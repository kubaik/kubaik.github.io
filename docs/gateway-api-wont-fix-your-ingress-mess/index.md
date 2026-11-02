# Gateway API won't fix your Ingress mess

## The stance: Gateway API is an ownership boundary, not an Ingress upgrade

The Kubernetes Gateway API tends to expose the difference between routing that works and routing that is trustworthy. It often looks fine in staging because staging never sees the traffic pattern that breaks it. This article covers the part that usually goes wrong.

The Gateway API is not a better Ingress. It is a different contract between the platform team and everyone else. Platform teams adopting it because Ingress is "deprecated" or "limited" risk inheriting a second, parallel configuration surface with its own CRDs, its own RBAC model, and its own upgrade treadmill — while the original Ingress objects stay exactly where they are, because removing them breaks every tenant who filed a ticket last quarter.

The contrarian position is simple: if the problem is that developers cannot self-serve HTTP routing safely, Gateway API is worth the migration cost. If the problem is that Ingress manifests are messy, Gateway API will make them messier, because it offers more knobs and more ways to misconfigure them. The API is not the fix; the ownership model it enables is. What follows is how to tell which situation applies before spending a quarter on a migration that changes nothing for users.

## Why "Ingress is deprecated, migrate" is the wrong reason

Ingress is not deprecated. The `networking.k8s.io/v1` Ingress API is stable and has been since Kubernetes 1.19. It is frozen in the sense that the community is not adding features to it, but frozen is not removed. The Gateway API project itself describes Ingress as continuing to be supported. Claims that Ingress is going away usually stem from confusion with the old `extensions/v1beta1` version that was removed, or from vendor material that wants customers on a newer controller.

The practical consequence: a migration driven by deprecation fear has no deadline and no forcing function. Teams start it, discover that every annotation on every Ingress maps to a different Gateway API filter or policy attachment, and stall halfway with both APIs live. That half-migrated state is worse than either endpoint, because there are now two routing planes, two sets of debugging tools, and two places a request can be misrouted.

The better reason to migrate is a capability gap you can name. Documented examples: Ingress has no first-class way to express header-based matching, traffic weighting across backends, or request mirroring without vendor annotations. Gateway API expresses these as part of the route spec. Ingress has no standard way to attach a policy (rate limiting, auth, TLS options) to a route without an annotation that only one controller understands. Gateway API has `BackendTLSPolicy`, `GRPCRoute`, and a policy attachment model that is still maturing but is at least a shared vocabulary. If the capability you need and the Ingress workaround you are tired of cannot be named, there is no migration reason yet.

## The mental model: three objects, three owners

What actually changes for platform teams is not the YAML. It is that the API forces an explicit record of who owns what. The model has three layers, and each maps to a team boundary:

| Object | Scope | Typical owner | What it answers |
|---|---|---|---|
| `GatewayClass` | Cluster | Infrastructure / platform | Which controller implements this class, and with what global config |
| `Gateway` | Namespaced (usually a platform namespace) | Platform team | Which listeners (ports, protocols, hostnames, TLS) are exposed, and which namespaces may attach routes |
| `HTTPRoute` (and `GRPCRoute`, `TLSRoute`) | Namespaced (tenant namespace) | Application team | How a specific host/path/header maps to backends |

The `Gateway` object has an `allowedRoutes` field on each listener that controls which namespaces can attach an `HTTPRoute` to it. That single field is the delegation mechanism. Set to `Same` (the default), only routes in the Gateway's own namespace can attach, and the result is Ingress with extra steps. Set to `All` or a label selector, and application teams can create routes in their own namespaces without a ticket for every new path. That is the operational change. Everything else is syntax.

A minimal, correct example that shows the boundary. The platform team creates the Gateway in a `platform-ingress` namespace:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: shared-gateway
  namespace: platform-ingress
spec:
  gatewayClassName: example-class
  listeners:
    - name: https
      protocol: HTTPS
      port: 443
      hostname: "*.example.com"
      tls:
        mode: Terminate
        certificateRefs:
          - kind: Secret
            name: wildcard-example-com
      allowedRoutes:
        namespaces:
          from: Selector
          selector:
            matchLabels:
              gateway-access: "true"
```

An application team in a namespace labeled `gateway-access: "true"` then creates its own route:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: checkout
  namespace: checkout-team
spec:
  parentRefs:
    - name: shared-gateway
      namespace: platform-ingress
  hostnames:
    - "checkout.example.com"
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /api
      backendRefs:
        - name: checkout-svc
          port: 8080
```

Note what the application team did not need: the platform team's namespace, the TLS secret, the listener port, or an annotation. Note what the platform team can still control: the hostname wildcard, the TLS termination mode, and which namespaces are allowed to attach at all. That is the contract Ingress never had.

## Where the old advice is still right

A few pieces of conventional wisdom survive the migration intact.

First, most teams should not run their own Gateway controller. The Gateway API is a spec; the implementation is a controller such as those from major cloud load balancers, ingress-nginx's successor work, or service mesh data planes. Running the controller is the same operational burden as running an Ingress controller, with the same upgrade and CVE cadence. If a cloud provider already supplies a managed load balancer, check whether its Gateway API implementation is generally available before building anything.

Second, the one-big-shared-Gateway pattern is still the right default for most clusters. The Gateway API supports many Gateways, but each listener consumes a port and a certificate, and a fleet of per-team Gateways is a fleet of load balancers. Start with one shared Gateway per environment and use `allowedRoutes` for delegation. Split only for a hard isolation requirement (separate TLS certs per business unit, separate IP allowlists, separate WAF policy).

Third, the debugging instinct is the same. When a route does not work, the question is still "did the controller see it, and did it accept it?" Gateway API improves this: every `Gateway` and `HTTPRoute` has a `status.parents` section with `conditions` that report whether the route was accepted, resolved, and programmed. That is more structured than reading controller logs to find out why an annotation was ignored, but it is the same investigation.

## The decision rule: delegate or wait

Use this rule, in order. It is deliberately blunt because the cost of a wrong migration is a quarter of platform time.

1. **Can you name the Ingress workaround you are tired of?** If the answer is "annotations are ugly," wait. If the answer is "canary weighting requires a vendor annotation that breaks on controller upgrade," there is a reason.
2. **Are application teams going to own `HTTPRoute` objects?** If the platform team will write every route anyway, Gateway API adds CRDs without adding delegation. Stay on Ingress until the ownership model changes.
3. **Does the chosen controller implement the needed features at the needed version?** The Gateway API has graduated features (the core `Gateway`, `HTTPRoute`, `GatewayClass` are GA) and experimental ones (`BackendTLSPolicy`, some filter types, `GRPCRoute` maturity varies by controller). Check the controller's conformance report before promising a capability to a tenant.
4. **Is there a rollback plan that keeps both APIs live?** During migration, a single hostname can be served by an Ingress and an `HTTPRoute` at the same time only if they point at different controllers or the same controller handles both. Plan per-hostname cutover, not big-bang.

If any of the first three answers is "no," the correct decision is to wait and revisit when it changes. That is not a failure; it is avoiding a migration that would have produced a second routing plane with no user-visible benefit.

## Failure modes that show up after the migration, not during it

The migration itself is mostly mechanical: translate host/path rules, replace annotations with filters or policy attachments, move TLS secrets to `certificateRefs`. The failures appear in the weeks after, when tenants start using the new delegation. Three are common enough to plan for.

**Route attachment sprawl.** Once application teams can create `HTTPRoute` objects, they will, and some will attach to the shared Gateway with a hostname that collides with another team's. Gateway API does not prevent two routes from claiming the same hostname and path; the controller picks a winner based on its own conflict resolution, and the loser gets a condition on `status.parents` that nobody reads. The fix is a policy, not a feature: either assign hostname prefixes per namespace via a label convention, or run an admission policy that rejects `HTTPRoute` objects whose `hostnames` do not match the namespace's assigned domain. Plan for this before flipping `allowedRoutes` to `All`.

**The `GatewayClass` becomes a shared dependency.** Every `Gateway` references a `GatewayClass`, and the `GatewayClass` carries controller-level configuration (which load balancer type, which global parameters). Changing it is a cluster-wide operation that can reprogram every listener at once. Treat `GatewayClass` changes like a cluster upgrade: staged, announced, and reversible. Teams that treat it as a per-team setting discover that one edit took down routing for every tenant.

**Status conditions are the new logs.** The structured `status.parents` conditions are better than annotation guessing, but only if tenants know to look. A route that is not accepted shows `Accepted: False` with a reason such as `NotAllowedByListeners` or `NoMatchingParent`. Platform teams should document the three or four conditions that matter and put them in the onboarding doc, because the failure mode is a developer staring at a 404 with no idea that their route was never attached.

## A worked example: measuring whether you actually need Gateway API

Rather than trusting a vendor's migration pitch, measure the gap. The following procedure takes about an hour per cluster and produces numbers you can defend in a planning meeting.

**Step 1: inventory annotation dependence.** List every Ingress and extract the annotations in use:

```bash
kubectl get ingress -A -o json |
  jq -r '.items[] | .metadata.namespace + "/" + .metadata.name as $n |
    (.metadata.annotations // {} | keys[]) | $n + " " + .'
```

Count how many distinct annotation keys appear and how many Ingress objects use each. An annotation that appears on one object is a local quirk; one that appears on dozens is a load-bearing behavior that must have a Gateway API equivalent before cutover.

**Step 2: classify each annotation.** Sort them into three buckets: (a) cosmetic or redundant with standard fields, (b) expressible as a core Gateway API filter such as `URLRewrite` or `RequestHeaderModifier`, (c) controller-specific policy with no portable equivalent. Bucket (c) is the migration risk. If bucket (c) is empty, the migration is largely mechanical. If it is large, the migration will either block on missing features or silently change behavior.

**Step 3: verify controller conformance.** Install the controller in a non-production cluster and read its conformance report for the exact version you intend to run. Then check the CRD versions present:

```bash
kubectl get crd gateways.gateway.networking.k8s.io \
  -o jsonpath='{.spec.versions[*].name}'
```

Compare the installed CRD versions against the `apiVersion` in your manifests. A mismatch produces objects that apply cleanly and are never reconciled, which is the most expensive kind of bug because nothing errors.

**Step 4: test delegation in a scratch namespace.** Create a namespace labeled `gateway-access: "true"`, bind a service account to a role that can create `httproutes`, and have a developer (not a platform engineer) create a route. Then inspect acceptance:

```bash
kubectl get httproute -n <namespace> -o yaml \
  | yq '.items[].status.parents'
```

If the conditions show `Accepted: True` and `ResolvedRefs: True`, delegation works end to end. If not, read the `reason` field — it names the exact listener or reference that failed, which is the whole point of the structured status.

The output of these four steps is a decision, not a feeling: a count of annotations that have no equivalent, a conformance gap list, and a demonstrated delegation path. Teams that skip this and migrate on schedule rather than on evidence tend to end up with both APIs live indefinitely.

## What to check before you commit

Before writing a single `HTTPRoute`, verify the controller, not the API. Gateway API conformance is reported per feature, and a controller that passes core conformance may still not implement the filter or policy being counted on. The upstream project publishes conformance reports; read the one for the controller version in use, not the marketing page. Then check two things in the cluster: whether the CRDs are installed at the version the manifests expect (the `kubectl get crd` command above), and whether RBAC already allows application-team service accounts to create `httproutes` in their namespaces. If RBAC is still platform-only, the object model exists without the delegation, and the migration will not change anything for developers.

## Frequently Asked Questions

**Is Ingress deprecated in Kubernetes?**
No. The `networking.k8s.io/v1` Ingress API is stable and remains supported. What happened is that the older `extensions/v1beta1` Ingress version was removed in Kubernetes 1.22, which is where a lot of the "Ingress is deprecated" talk originates. The Gateway API project positions itself as the successor for new feature work, but Ingress objects continue to function and controllers continue to reconcile them.

**Can Ingress and Gateway API run at the same time?**
Yes, and most migrations do. They are separate APIs reconciled by controllers, and a cluster can have an Ingress controller and a Gateway controller running side by side. The risk is not technical conflict but operational confusion: two routing planes mean two places to debug a 404. Plan a per-hostname cutover so each hostname has exactly one authoritative routing object at any time.

**Do application teams need cluster-admin to create HTTPRoutes?**
No. `HTTPRoute` is namespaced, so a role binding in the team's namespace is enough, provided the `Gateway` listener's `allowedRoutes` permits that namespace to attach. The platform team controls attachment via the Gateway; the application team controls routing within the hosts and paths the Gateway exposes. That separation is the point of the API.

**What replaces Ingress annotations like rewrite-target or rate limiting?**
Rewrites are expressed as `URLRewrite` filters on `HTTPRoute` rules. Rate limiting and auth are not core Gateway API fields; they are attached through the policy attachment model, and support depends on the controller. This is where "Gateway API is portable" is weakest in practice: the core routing spec is portable, but policy attachments are still controller-specific. Check the controller's documentation before assuming a policy relied on today has a direct equivalent.

## The 30-minute check

Open the current Ingress manifests, run the annotation inventory command from the worked example, and count how many objects rely on a controller-specific annotation for behavior that would be lost on migration. Then run `kubectl get gatewayclass` in a cluster where the chosen controller is installed and read the conformance report for that controller version. If the annotation count is low and the conformance report covers the needed features, there is a migration case; if the annotation count is high and the report is thin, there is a documented reason to wait.
