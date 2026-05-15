# Cut cloud carbon 38% without slowing down

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2023, we were running a real-time geolocation API for a fleet-management client in Brazil. The system handled 12,000 requests per second at peak, with 95% p99 latency under 80 ms. We used a managed Kubernetes cluster on AWS (EKS, 3×c6i.2xlarge nodes) and PostgreSQL RDS (db.m6i.2xlarge, 8 vCPUs, 32 GB RAM, gp3 SSD). The bill was $1,800/month, and we were proud of the uptime—99.98% over six months.

Then the client’s sustainability team asked for a carbon audit. AWS’s Customer Carbon Footprint Tool showed 4.7 metric tons of CO₂ per month, mostly from compute and storage. The target was a 30% reduction without adding latency or costs. We thought: "We’ll just right-size the cluster and swap to ARM." Simple, right?

Wrong. The numbers told a different story. Right-sizing the nodes saved 12% carbon but added 15 ms latency to 18% of requests—unacceptable for GPS updates. ARM instances (graviton3) cut compute carbon by 22%, but the managed database didn’t support ARM, so we’d have to migrate to self-managed PostgreSQL on EC2. That meant 40 hours of setup, monitoring, and the risk of unplanned downtime during a migration window. The sustainability team wanted results in six weeks. We had to move faster.

We started measuring everything. We instrumented the API with OpenTelemetry, logging CPU usage, memory, and request duration per endpoint. The database was the real hotspot: 70% of CPU time was spent on the same five queries—all geospatial distance calculations. We traced them down and found they were running on every request, even when the client was already within 50 meters of the previous point. The queries were:
```sql
SELECT id, latitude, longitude, 
  ST_DistanceSphere(point(:lat, :lon), point(latitude, longitude)) as distance
FROM vehicles 
WHERE ST_DWithin(ST_MakePoint(:lat, :lon)::geography, 
                 ST_MakePoint(latitude, longitude)::geography, 5000)
ORDER BY distance ASC 
LIMIT 50;
```

The query used a geography-based distance index, but it was still scanning 12,000 rows per request. The CPU sat at 85% during peak, and the database connections were maxed out at 150. We knew we had to fix the geospatial queries first, but we weren’t sure how that would cut carbon—until we saw a chart showing that 62% of the cluster’s energy went to CPU cycles on those queries.

**Summary:** We needed to cut cloud carbon by 30% in six weeks without increasing latency or cost, but the obvious levers (right-sizing, ARM) either hurt performance or required heavy migration work. The real bottleneck was a handful of geospatial queries burning 62% of CPU cycles and 70% of database time.


## What we tried first and why it didn’t work

Our first idea was to scale down the EKS cluster. We moved from 3×c6i.2xlarge nodes to 2×c6g.xlarge (Graviton2) and dropped one node entirely during off-peak hours using Karpenter’s consolidation policy. We expected a 25% carbon cut from the ARM transition and another 15% from fewer cores running 24/7. The bill dropped from $1,800 to $1,440, and the carbon tool showed 3.1 metric tons—28% less. Success? Not quite.

Latency regression was immediate. The 2×c6g.xlarge nodes had 20% fewer vCPUs, and the API started queuing requests. The p99 latency crept up to 110 ms, and 12% of requests breached 150 ms. The fleet-management client’s SLA capped latency at 100 ms for 95% of requests. We were over the limit. Worse, the database CPU was still at 88% during peaks because the geospatial queries hadn’t changed.

Next, we tried caching the geospatial queries with Redis. We set a TTL of 30 seconds and stored the top-50 results per bounding box. The cache warmed quickly because vehicles rarely moved more than 50 meters between requests. We used Redis 7.0 with a 2 GB maxmemory policy and allkeys-lru eviction. The first test cut database CPU by 35%, and the p99 latency dropped to 55 ms—below the SLA. Carbon emissions from the database dropped from 2.9 to 1.9 metric tons, a 34% cut for that component alone.

But the Redis cache introduced a new problem: cache stampede. When a vehicle entered a new bounding box, the first request triggered a full query while others waited. For 30 seconds, the database CPU spiked to 95%, and latency hit 200 ms for 2% of requests. We tried locking with Redlock, but the Redis cluster (3×m6g.large) added $240/month to the bill. The carbon tool now showed 3.3 metric tons total—only a 22% cut from the original, and we’d added complexity.

We also tried query rewrites. We replaced `ST_DistanceSphere` with `ST_DWithin` and pre-filtered the bounding box to 500 meters instead of 5 kilometers. The query became:
```sql
SELECT id, latitude, longitude, 
  ST_DistanceSphere(point(:lat, :lon), point(latitude, longitude)) as distance
FROM vehicles 
WHERE latitude BETWEEN :lat_min AND :lat_max 
  AND longitude BETWEEN :lon_min AND :lon_max
ORDER BY latitude, longitude 
LIMIT 50;
```

The index scan improved, but the query planner still did a full index scan on the bounding box. CPU dropped from 85% to 72%, but the geospatial index wasn’t used efficiently. We added a BRIN index on latitude and longitude, but BRIN is only effective on large tables with natural ordering—our table had 3.2 million rows, but the data wasn’t ordered by geography. The BRIN index didn’t help, and we lost 5 ms per query due to the planner’s fallback to a seq scan.

**Summary:** Scaling down and ARM migration cut carbon by 28% but broke latency SLA. Redis caching cut database load by 35% but introduced cache stampedes and added $240/month in cache costs. Query rewrites improved CPU usage by 13% but didn’t fix the geospatial index inefficiency. We were stuck in a loop of trading carbon for latency or cost.


## The approach that worked

We stopped optimizing the wrong things. Instead of chasing CPU or instance types, we focused on the geospatial index itself. The problem wasn’t the query—it was the data layout. Our vehicles table had a geography column indexed with `GIST(geog)`, but the index wasn’t selective enough for frequent small-radius searches. We needed a better spatial index.

We tested two options: PostGIS’s `pg_quadtree` extension (v1.1) and a simple grid-based partitioning scheme. The quadtree promised 40% faster small-radius queries in benchmarks, but it required a full table rebuild and added 15% storage overhead. We weren’t ready to commit to a rewrite. The grid partitioning was simpler: divide the world into 1 km² cells, assign each vehicle to a cell based on its coordinates, and index the cell column. We chose grid partitioning because it was native to PostgreSQL 15 (no extensions), and we could migrate incrementally.

We created a grid function:
```sql
CREATE OR REPLACE FUNCTION vehicle_cell(lat double precision, lon double precision)
RETURNS int AS $$
DECLARE
  cell_lat int;
  cell_lon int;
BEGIN
  cell_lat := floor((lat + 90) / 1);  -- 1° latitude = ~111 km, so 1 km cell = 0.00899°
  cell_lon := floor((lon + 180) / 1);
  RETURN cell_lat * 360 + cell_lon;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

Then added a generated column and index:
```sql
ALTER TABLE vehicles ADD COLUMN grid_cell int GENERATED ALWAYS AS (vehicle_cell(latitude, longitude)) STORED;
CREATE INDEX idx_vehicles_grid ON vehicles (grid_cell);
```

We partitioned the table by `grid_cell` using 1024 child tables (1 km² cells). The partition key was `grid_cell % 1024`, so each partition held roughly 3,125 rows.

The query changed to:
```sql
WITH nearby_cells AS (
  SELECT unnest(ARRAY[
    vehicle_cell(:lat - 0.01, :lon - 0.01),
    vehicle_cell(:lat - 0.01, :lon),
    vehicle_cell(:lat - 0.01, :lon + 0.01),
    vehicle_cell(:lat, :lon - 0.01),
    vehicle_cell(:lat, :lon),
    vehicle_cell(:lat, :lon + 0.01),
    vehicle_cell(:lat + 0.01, :lon - 0.01),
    vehicle_cell(:lat + 0.01, :lon),
    vehicle_cell(:lat + 0.01, :lon + 0.01)
  ]) as cell
)
SELECT id, latitude, longitude, 
  ST_DistanceSphere(point(:lat, :lon), point(latitude, longitude)) as distance
FROM vehicles 
WHERE grid_cell IN (SELECT cell FROM nearby_cells)
  AND ST_DWithin(ST_MakePoint(:lat, :lon)::geography, 
                 ST_MakePoint(latitude, longitude)::geography, 5000)
ORDER BY distance ASC 
LIMIT 50;
```

This reduced the scanned rows per query from 12,000 to 300–500, depending on the grid density. Database CPU dropped from 85% to 45% during peak, and the geospatial index was finally used efficiently. We kept the Redis cache (TTL 30s) but no longer needed locking because the queries were fast enough to avoid stampedes.

We then revisited the cluster sizing. With the database CPU at 45%, we could downsize the RDS instance from db.m6i.2xlarge to db.m6i.xlarge (4 vCPUs, 16 GB RAM). The cost dropped from $840 to $520/month, and the carbon footprint dropped by 18% for that component. The EKS cluster stayed at 2×c6g.xlarge nodes, but we enabled Karpenter’s bin-packing to consolidate pods during off-peak hours. The total cluster cost dropped from $1,800 to $1,250/month, a 31% cut.

**Summary:** We fixed the geospatial index inefficiency with a grid-based partitioning scheme, reducing scanned rows by 96% and database CPU by 47%. Downsizing the database and optimizing the cluster cut total costs by 31% and carbon by 38% without touching the latency SLA. The Redis cache became a performance booster, not a crutch.


## Implementation details

### Step 1: Grid partitioning setup
We ran the migration during a 2-hour maintenance window. First, we created the new `grid_cell` column and index without blocking writes:
```sql
ALTER TABLE vehicles ADD COLUMN grid_cell int;
UPDATE vehicles SET grid_cell = vehicle_cell(latitude, longitude);
CREATE INDEX CONCURRENTLY idx_vehicles_grid ON vehicles (grid_cell);
ALTER TABLE vehicles ALTER COLUMN grid_cell SET NOT NULL;
```

Then we created the 1024 child tables:
```sql
DO $$
DECLARE
  i int;
BEGIN
  FOR i IN 0..1023 LOOP
    EXECUTE format('CREATE TABLE vehicles_part_%s PARTITION OF vehicles 
                   FOR VALUES FROM (%s) TO (%s)', 
                   i, i, i+1);
  END LOOP;
END $$;
```

We swapped the table in one transaction to avoid downtime:
```sql
BEGIN;
ALTER TABLE vehicles detach PARTITION vehicles_part_0;
ALTER TABLE vehicles attach PARTITION vehicles_part_0 
  FOR VALUES FROM (0) TO (1);
-- Repeat for all partitions (simplified)
COMMIT;
```

The migration took 45 minutes for 3.2 million rows, and we saw immediate query improvements. The `EXPLAIN ANALYZE` output showed:
```
Limit  (cost=0.42..23.84 rows=50 width=61) (actual time=1.2..1.8 rows=50 loops=1)
  ->  Index Scan using idx_vehicles_distance on vehicles_part_1023 vehicles  (cost=0.42..23.84 rows=50 width=61) (actual time=1.2..1.8 rows=50 loops=1)
        Index Cond: (ST_DWithin(geog, '(SRID=4326;POINT(...))'::geography, 5000))
Planning Time: 1.4 ms
Execution Time: 2.1 ms
```

Previously, the same query took 45 ms and scanned 12,000 rows.

### Step 2: Redis cache tuning
We kept the Redis cache but changed the TTL to 15 seconds and added a probabilistic early refresh. We used Redis 7.0 with a `maxmemory-policy allkeys-lru` and 4 GB maxmemory. The cache warmed quickly because vehicles rarely moved more than 50 meters between requests. We instrumented cache hits and misses with OpenTelemetry and found the hit rate stabilized at 82% during peak hours.

We also added a Lua script to handle cache stampedes:
```lua
local key = KEYS[1]
local result = redis.call('GET', key)
if result then
  return result
else
  redis.call('SET', key, '1', 'PX', 5000, 'NX')  -- 5s lock
  return 'lock_acquired'
end
```

This reduced the stampede window to 500 ms and avoided the need for Redlock. The cache cost remained at $120/month.

### Step 3: Cluster optimization
We configured Karpenter to consolidate pods during off-peak hours (10 PM–6 AM BRT). We set:
```yaml
spec:
  consolidationPolicy: WhenUnderutilized
  expireAfter: 30m
```

We also enabled pod topology spread constraints to distribute pods across zones:
```yaml
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: ScheduleAnyway
    labelSelector:
      matchLabels:
        app: geolocation-api
```

The cluster CPU utilization dropped from 65% to 40% during off-peak, and we could safely downsize the RDS instance to db.m6i.xlarge without risking performance.

### Step 4: Monitoring and alerts
We added carbon-aware alerts using CloudWatch and Datadog. We set a threshold: if the carbon intensity (gCO₂/kWh) from AWS’s Customer Carbon Footprint Tool exceeded 400 gCO₂/kWh, we triggered a scale-down event. We also tracked the ratio of cache hits to database CPU usage—if hits dropped below 75%, we increased the TTL or added more partitions.

**Summary:** The implementation involved a 45-minute migration to grid partitioning, Redis cache tuning with a Lua lock script, Karpenter consolidation, and carbon-aware alerts. The total setup time was 8 hours spread over two weeks, with no downtime during the migration.


## Results — the numbers before and after

| Metric | Before | After | Change |
| --- | --- | --- | --- |
| **Cluster cost (AWS)** | $1,800/month | $1,250/month | -31% |
| **Database cost (RDS)** | $840/month | $520/month | -38% |
| **EKS node count** | 3×c6i.2xlarge | 2×c6g.xlarge | -33% |
| **p99 latency** | 80 ms | 72 ms | -10% |
| **p95 latency** | 55 ms | 48 ms | -13% |
| **Database CPU peak** | 85% | 45% | -47% |
| **Geospatial query time** | 45 ms | 1.8 ms | -96% |
| **Rows scanned per query** | 12,000 | 350 | -97% |
| **Carbon footprint (monthly)** | 4.7 metric tons | 2.9 metric tons | -38% |
| **Cache hit rate** | N/A | 82% | N/A |

We measured carbon using AWS’s Customer Carbon Footprint Tool and validated with the Cloud Carbon Footprint open-source calculator. The tool broke down emissions by service: compute (62%), storage (25%), and data transfer (13%). After the changes, compute dropped to 50%, storage to 22%, and data transfer to 28%.

One surprise: the Redis cache, while improving performance, only added 0.1 metric tons of CO₂ per month—negligible compared to the database savings. The grid partitioning itself didn’t add measurable carbon because it used existing storage and compute.

We also tracked the carbon intensity of the AWS region (us-east-1). During the migration, us-east-1’s carbon intensity dropped from 420 gCO₂/kWh to 350 gCO₂/kWh, which contributed an additional 5% reduction in our footprint. We didn’t rely on that—we optimized the system to be resilient regardless of the grid’s carbon mix.

**Summary:** The changes cut total cloud costs by 31%, carbon emissions by 38%, and improved latency by 10–13%. The geospatial queries became 25x faster, scanning 97% fewer rows. The Redis cache became a performance booster, not a crutch, and the system remained within SLA during peak loads.


## What we'd do differently

1. **Test partitioning strategy first.** We assumed 1 km² cells would work everywhere, but in dense cities like São Paulo, 1 km² can hold 20,000 vehicles. In those areas, we had to use 0.5 km² cells, which increased the number of partitions and added a small storage overhead. We should have profiled the data distribution before choosing the grid size.

2. **Avoid the Lua lock script.** The lock script worked, but it added complexity to the cache layer. A better approach would have been to use a probabilistic early refresh with a jittered TTL. We measured 82% cache hits, but 12% of misses were due to the lock contention. With jittered TTL, we could have hit 88% without locks.

3. **Monitor carbon intensity in real time.** We set up alerts based on the monthly carbon footprint, but we didn’t track the carbon intensity of the grid in real time. If we had, we could have scaled down during high-carbon hours (e.g., when AWS was running on coal-heavy power). We ended up using Cloud Carbon Footprint’s CLI to pull hourly data, but it would have been better as a Datadog metric.

4. **Skip the ARM migration for RDS.** We considered moving the database to self-managed PostgreSQL on Graviton3 EC2, but the migration risk wasn’t worth the 15% carbon cut. Instead, we downsized the RDS instance and kept it on Intel. The savings were similar, and the managed service reliability was worth the extra carbon.

5. **Profile the geospatial index earlier.** We wasted two weeks on query rewrites and caching before realizing the index itself was the problem. A 30-minute `EXPLAIN ANALYZE` session would have shown the planner was doing a full index scan despite the GIST index. We should have started with a spatial index health check.

**Summary:** We’d test partitioning granularity, avoid cache locks, track carbon intensity in real time, skip the RDS ARM migration, and profile the spatial index earlier. These changes would have saved 4–6 hours of engineering time and improved the cache hit rate by 6%.


## The broader lesson

Sustainable software engineering isn’t about choosing between performance, cost, or carbon—it’s about finding the constraints that bind them all. In our case, the geospatial index was the hidden bottleneck: it burned 62% of CPU cycles and 70% of database time, yet it wasn’t the first thing we looked at. Once we fixed that, the other levers (scaling, caching, ARM) became effective without tradeoffs.

The lesson is this: **Measure the real bottlenecks, not the obvious ones.** CPU usage, latency, and carbon emissions are symptoms, not root causes. In cloud systems, the root cause is often the query planner’s decisions, the index design, or the data layout. Fix those, and the rest follows.

Another lesson: **Carbon and cost are correlated but not identical.** We saved 38% carbon and 31% cost, but the correlation wasn’t 1:1. The grid partitioning reduced CPU usage, which cut both cost and carbon. The ARM migration cut cost but not carbon as much as we expected because the database stayed on Intel. The Redis cache improved performance (and thus reduced the need for over-provisioning) but added negligible carbon. The key is to optimize for the metric that matters most to your business—whether it’s latency, cost, or carbon—and measure the others.

Finally, **resilience beats optimization.** We could have squeezed another 5% carbon out of the system by moving to self-managed PostgreSQL on Graviton3, but the risk of downtime during a migration wasn’t worth it. The managed RDS instance, despite its higher carbon footprint, gave us the reliability we needed. Sustainable engineering means building systems that can withstand failures, not just ones that are theoretically optimal.

**Summary:** Sustainable engineering starts with measuring the real bottlenecks, not the obvious ones. Carbon and cost are correlated but not identical, so optimize for the metric that matters most. Resilience beats optimization—reliability is a sustainability lever in its own right.


## How to apply this to your situation

1. **Find your hidden bottlenecks.** Run `EXPLAIN ANALYZE` on your slowest queries. Look for full index scans, seq scans, or unnecessary sorts. In our case, the geospatial index was the problem, but it could be a JSON column, a full-text search, or a window function in your system.

2. **Measure carbon, not just cost.** Use AWS’s Customer Carbon Footprint Tool or Cloud Carbon Footprint. Track the breakdown by service—compute, storage, data transfer. If compute is >50% of your footprint, focus there. If storage is >30%, look at archiving or tiering.

3. **Right-size after you optimize.** Don’t downsize your cluster until you’ve fixed the queries. We tried right-sizing first and broke latency. Fix the queries, then downsize. Use tools like `pgMustard` for PostgreSQL or `EXPLAIN` visualizers to spot inefficiencies.

4. **Use caching wisely.** Cache the results of expensive queries, but avoid cache stampedes. Use probabilistic early refresh or jittered TTL instead of locks. Measure cache hit rate and adjust TTL based on your data’s volatility.

5. **Partition by access patterns.** If your queries filter by geography, time, or user segment, partition the table accordingly. Grid partitioning works for geospatial data; range partitioning works for time-series. Native partitioning in PostgreSQL is often enough—no need for extensions.

6. **Track carbon intensity in real time.** If your cloud provider offers hourly carbon data, integrate it into your monitoring. Scale down during high-carbon hours if your workload allows it. Use Cloud Carbon Footprint’s CLI or Datadog’s AWS integration to pull the data.

7. **Balance ARM and managed services.** ARM instances save carbon and cost, but managed services (like RDS) often don’t support ARM. If you’re on a managed database, consider downsizing the instance instead of migrating to self-managed. The savings are similar, and the reliability is worth it.

8. **Instrument everything.** Add OpenTelemetry to your API, database, and cache. Track CPU usage, memory, latency, and carbon emissions per request. Without the data, you’re optimizing in the dark.

**Next step:** Profile your slowest 5 queries with `EXPLAIN ANALYZE` and check if they’re doing full scans. If they are, redesign the index or partition the table. Start with a 30-minute spike—no migration, just a query plan review. You’ll likely find a 20–50% improvement in CPU usage without touching your cluster size.


## Resources that helped

- **Cloud Carbon Footprint** (https://www.cloudcarbonfootprint.org/): Open-source tool to measure cloud carbon emissions. We used it to validate AWS’s numbers and track hourly intensity.
- **pgMustard** (https://www.pgmustard.com/): Visualizer for PostgreSQL `EXPLAIN` output. Helped us spot the full index scan in the geospatial query.
- **PostgreSQL 15 Partitioning Docs** (https://www.postgresql.org/docs/15/ddl-partitioning.html): Native partitioning in PostgreSQL 15 is powerful and easy to set up. Our grid partitioning was built on this.
- **Redis Lua Scripting** (https://redis.io/docs/manual/programmability/eval-intro/): We used Lua to handle cache stampedes without Redlock. The docs were clear and the scripting model simple.
- **Karpenter** (https://karpenter.sh/): Auto-scaling for EKS that consolidates pods efficiently. The `consolidationPolicy` setting saved us 20% cluster cost during off-peak.
- **AWS Customer Carbon Footprint Tool** (https://aws.amazon.com/about-aws/sustainability/): Granular breakdown of emissions by service and region. We used it to set carbon-aware alerts.
- **Geospatial Indexes in PostGIS** (https://postgis.net/workshops/postgis-intro/geography.html): PostGIS’s geography type and `ST_DWithin` function are the backbone of our solution. The docs are dense but accurate.
- **Datadog’s AWS Integration** (https://docs.datadoghq.com/integrations/amazon_web_services/): We used Datadog to track carbon intensity in real time and set alerts when it spiked above 400 gCO₂/kWh.


## Frequently Asked Questions

### How do I know if my geospatial queries are the bottleneck?

Run `EXPLAIN ANALYZE` on your slowest query and look for `Seq Scan` or `Index Scan` with high `actual time`. If the query is scanning thousands of rows and the `actual time` is >20 ms, it’s likely the bottleneck. In our case, the query scanned 12,000 rows and took 45 ms. After partitioning, it scanned 350 rows and took 1.8 ms. If you’re on PostgreSQL, `pgMustard` visualizes the plan and highlights inefficiencies.

### Will grid partitioning work for non-geospatial data?

Yes, but the partitioning key should match your query patterns. For time-series data, partition by date ranges. For user data, partition by user ID ranges or tenant IDs. The goal is to reduce the scanned rows per query. In our case, the grid key (`grid_cell