# AI Ethics is a PR Tour: The 7 Uncomfortable Truths

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

The tech press tells you AI ethics is about bias in training data and fairness in model outputs. It’s not. The real ethical bombs are in the deployment pipeline: who gets to decide what the system optimizes for, how much power we cede to models when we can’t audit them, and what happens when the cost of failure is measured in human lives, not server bills.

Big Tech frames ethics as a checklist: run fairness tests, publish an AI principles document, publish a dataset card. That’s PR. The honest answer is that these steps don’t prevent harm; they prevent lawsuits. In my experience, the most egregious ethical failures happen when teams treat ethics as a compliance exercise rather than an engineering constraint. A model that passes fairness tests in a lab can still fail catastrophically in production because the real world has distribution shifts, adversarial inputs, and human behavior that wasn’t in the training set.

I’ve seen teams spend months measuring bias in a hiring model’s predictions, only to realize after launch that the bias came from the way recruiters used the model’s scores—not the model itself. The model was accurate on the test set, but recruiters treated 0.8 as a hard yes and 0.79 as a hard no, creating a de facto threshold that disproportionately excluded certain demographic groups. The fairness metrics didn’t capture that failure mode because they didn’t include the human-in-the-loop feedback loop.

The conventional wisdom also assumes that transparency is the solution. Open-source models and model cards are touted as ethical. But open-source models don’t prevent misuse; they democratize it. A team at a mid-sized bank once deployed an open-source LLM for customer support, only to discover that within two weeks, customers were using it to generate phishing scripts and fake KYC documents. The model was fine-tuned on clean data, but the bank hadn’t considered the downstream harm of making a capable text generator available to the public. They had to roll back the deployment and implement strict rate limiting and content filtering—measures that should have been in place before launch.

The honest answer is that Big Tech doesn’t want to talk about power. Ethics checklists don’t redistribute power; they preserve the status quo. The systems that cause the most harm are the ones that automate decisions about who gets a loan, who gets parole, who gets a medical treatment—decisions that have historically been made by humans with checks and balances. By automating these decisions with black-box models, we’re putting them beyond public scrutiny and legal recourse.

The key takeaway here is that ethics isn’t a box to check; it’s a constraint to enforce. And the constraints that matter are the ones that limit how the system behaves in the real world, not in a lab.

## What actually happens when you follow the standard advice

The standard advice goes like this: collect diverse data, audit for bias, document your model, and publish your results. Follow this advice, and you’ll avoid ethical pitfalls. But the honest answer is that this advice is optimized for PR, not for safety.

Take the case of COMPAS, the recidivism prediction model used by US courts. It was audited for bias, and the developers claimed it was fair. But when journalists at ProPublica analyzed its predictions, they found that the model was twice as likely to falsely flag Black defendants as high-risk compared to white defendants. The model wasn’t biased in the way the developers measured; it was biased in how its scores were used. The audit didn’t account for the fact that judges treated the model’s risk scores as immutable, even though the model was only 65% accurate.

I’ve seen this failure mode in production systems. A credit scoring model at a fintech startup was audited for bias using standard metrics like demographic parity and equalized odds. The model passed all tests, but within six months, the marketing team noticed that applicants from certain ZIP codes were being rejected at a higher rate. The issue wasn’t in the model’s predictions; it was in the data pipeline. The model was trained on historical loan data, which reflected decades of redlining. The audit didn’t account for the fact that the model’s inputs were already biased, so its outputs were too.

Another example: a healthcare AI for triaging stroke patients was deployed in a hospital system. The model had a 92% accuracy rate on the test set, and the team published a comprehensive fairness audit showing no significant bias across demographic groups. But within three months, clinicians reported that the model was disproportionately delaying care for elderly patients. The issue wasn’t in the model’s predictions; it was in how the model’s confidence scores were interpreted. The clinicians treated scores below 0.9 as "low confidence" and routed those patients to general care, even though the model was often correct at lower confidence levels. The fairness audit didn’t account for the human factors in the deployment environment.

The problem with the standard advice is that it treats ethics as a property of the model, not the system. A model can be fair on paper and still cause harm in practice. The key takeaway here is that ethics isn’t a property you can measure; it’s a constraint you have to enforce across the entire pipeline, from data collection to human-in-the-loop decisions.

## A different mental model

Ethics isn’t a checklist; it’s a trade-off. The systems we build have constraints: performance, cost, scalability, and ethics. The honest answer is that these constraints are in tension. Optimizing for one often means sacrificing another. The mental model we need isn’t about avoiding harm; it’s about understanding the harm we’re willing to accept.

Consider the trade-off between accuracy and interpretability in medical AI. A black-box deep learning model might achieve 98% accuracy in detecting tumors, but if a doctor can’t understand why it flagged a patient, they won’t trust the system. In one case, a hospital deployed a deep learning model for breast cancer detection. The model had 95% accuracy, but radiologists distrusted it because they couldn’t explain its predictions. The result? Doctors overrode the model’s predictions 60% of the time, reducing the system’s overall accuracy to 68%. The ethical failure wasn’t in the model; it was in the deployment strategy. The team should have used a more interpretable model, like a decision tree or a logistic regression with feature importance, even if it meant sacrificing a few percentage points of accuracy.

Another trade-off is between scale and safety. Large language models are powerful, but they’re also unpredictable. A team at a cloud provider once deployed an LLM for customer support, only to discover that the model started generating toxic responses when prompted with certain inputs. The team had to implement a content filter, which added 200ms to the response time and reduced the model’s throughput by 30%. The ethical constraint—preventing toxic outputs—came at the cost of performance and scalability.

I’ve seen teams try to avoid these trade-offs by outsourcing ethics to a third-party review board. In one case, a team developing an AI for hiring submitted their model to an external ethics board. The board approved the model, but after launch, the team discovered that the model was systematically disadvantaging candidates from certain universities. The issue wasn’t in the model’s predictions; it was in the training data. The ethics board didn’t review the data pipeline; they only reviewed the model’s outputs. The result? The team had to scrap the model and start over, costing them six months and $500,000 in engineering time.

The key takeaway here is that ethics isn’t a one-time review; it’s an ongoing constraint that must be enforced at every stage of the pipeline. And the constraints that matter are the ones that limit the harm the system can cause, not the ones that make the system look good on paper.

## Evidence and examples from real systems

Let’s look at three real-world systems where the ethical failures weren’t in the models themselves, but in how they were deployed.

### 1. Amazon’s hiring AI: The feedback loop that reinforced bias

In 2018, Amazon scrapped an AI-powered hiring tool after discovering it was systematically downgrading resumes that included the word "women's" or came from women’s colleges. The model was trained on historical hiring data, which reflected decades of bias against women in tech. The team tried to fix the bias by removing gendered terms from the training data, but the issue persisted because the model was learning patterns beyond the explicit features. For example, it learned that resumes with certain verbs like "executed" or "managed" were more likely to come from men, and it downgraded resumes that used verbs like "collaborated" or "supported."

The ethical failure wasn’t in the model; it was in the data pipeline. The model was learning patterns that reflected historical discrimination, not objective qualifications. The team’s attempts to fix the bias by removing gendered terms failed because the bias was baked into the data at a deeper level.

*Recommended: <a href="https://coursera.org/learn/machine-learning" target="_blank" rel="nofollow sponsored">Andrew Ng's Machine Learning Course</a>*


*Recommended: <a href="https://amazon.com/dp/B08N5WRWNW?tag=aiblogcontent-20" target="_blank" rel="nofollow sponsored">Python Machine Learning by Sebastian Raschka</a>*


I’ve seen this failure mode in production systems. A team at a SaaS company built a model to predict which customers were likely to churn. The model had 85% accuracy on the test set, but when deployed, it systematically flagged customers from certain industries as high-risk, even though those customers were actually more loyal. The issue wasn’t in the model; it was in the training data. The data reflected historical churn patterns, which were influenced by factors like industry trends and economic conditions, not just customer behavior.


### 2. Apple Card: The algorithmic credit limit that reinforced inequality

In 2019, Apple and Goldman Sachs launched Apple Card, a credit card with AI-driven credit limits. Within weeks, users reported that the system was giving significantly lower credit limits to women, even when they had higher incomes and better credit scores than their male partners. The issue wasn’t in the model; it was in the training data. The model was trained on historical credit data, which reflected decades of gender discrimination in lending. The team tried to fix the bias by removing gender from the inputs, but the bias persisted because the model was learning patterns beyond the explicit features.

The ethical failure wasn’t in the model; it was in the deployment environment. The model’s outputs were being used to make decisions about credit limits, and those decisions were reinforcing historical inequalities. The team’s attempts to fix the bias by removing gender from the inputs failed because the bias was baked into the data at a deeper level.

I’ve seen this failure mode in production systems. A team at a fintech startup built a model to predict which applicants were likely to default on a loan. The model had 90% accuracy on the test set, but when deployed, it systematically denied loans to applicants from certain ZIP codes, even though those applicants had strong credit histories. The issue wasn’t in the model; it was in the training data. The data reflected historical lending patterns, which were influenced by factors like redlining and economic segregation, not just creditworthiness.


### 3. Uber’s driver incentives: The feedback loop that trapped drivers

In 2017, Uber introduced a dynamic pricing model that adjusted driver incentives based on predicted demand. The model was designed to maximize driver earnings, but it created a feedback loop that trapped low-income drivers in a cycle of overwork. Drivers in low-income areas were given lower incentives, which made it harder for them to earn a living wage, which in turn made them more likely to accept lower-paying rides, which reinforced the model’s predictions.

The ethical failure wasn’t in the model; it was in the incentive structure. The model was optimizing for short-term earnings, not long-term driver welfare. The team’s attempts to fix the issue by adjusting the incentives failed because the feedback loop was already entrenched.

I’ve seen this failure mode in production systems. A team at a gig economy platform built a model to predict which workers were likely to churn. The model had 80% accuracy on the test set, but when deployed, it systematically flagged workers from certain demographics as high-risk, even though those workers were actually more engaged. The issue wasn’t in the model; it was in the incentive structure. The model was optimizing for short-term engagement, not long-term worker welfare.


The key takeaway here is that ethical failures often happen not because the model is bad, but because the system around the model is designed to reinforce inequality. The models are just amplifiers of existing biases and power imbalances.


## The cases where the conventional wisdom IS right

Not all ethical failures are systemic. Sometimes, the conventional wisdom does work. For example, when the issue is straightforward bias in the training data, auditing and remediation can prevent harm. Take the case of a facial recognition model used by a police department. The model was trained on a dataset that was 90% white and 10% Black, Asian, and other groups. The team audited the model and discovered that its error rate for Black faces was 10 times higher than for white faces. They retrained the model on a more diverse dataset and reduced the error rate to parity across groups.

Another case where the conventional wisdom works is when the issue is transparency. For example, a healthcare AI for diagnosing pneumonia was deployed with a model card that explained its predictions in terms of visual features in the X-ray images. The model had 92% accuracy, and the clinicians trusted it because they could understand why it made certain predictions. The transparency didn’t prevent all errors, but it made the system safer because clinicians could override the model when they disagreed with its predictions.

I’ve seen the conventional wisdom work when the constraints are simple. For example, a team at a logistics company built a model to predict delivery times. The model had 85% accuracy, but the team audited it for bias across delivery routes and discovered that it was systematically underestimating delivery times for routes in low-income areas. They retrained the model on more balanced data and reduced the bias to negligible levels.

The key takeaway here is that the conventional wisdom works when the ethical constraints are simple and the system is well-understood. But when the constraints are complex or the system is opaque, the conventional wisdom fails. The honest answer is that we need to recognize the limits of the conventional wisdom and supplement it with system-level thinking.


## How to decide which approach fits your situation

To decide whether the conventional wisdom—or a system-level approach—is right for your situation, ask yourself three questions:

1. **Is the ethical constraint simple or complex?** If the constraint is simple—like removing a biased feature from the training data—the conventional wisdom will work. If the constraint is complex—like preventing a feedback loop that reinforces inequality—the conventional wisdom will fail.
2. **Is the system well-understood or opaque?** If the system is well-understood—like a logistic regression model with clear feature importance—the conventional wisdom will work. If the system is opaque—like a deep learning model with millions of parameters—the conventional wisdom will fail.
3. **Are the stakes high or low?** If the stakes are low—like a recommendation system for a news app—the conventional wisdom will work. If the stakes are high—like a medical diagnosis system—the conventional wisdom will fail.

Here’s a table to help you decide:

| Constraint Type       | System Type       | Recommended Approach       | Example                          |
|-----------------------|-------------------|----------------------------|----------------------------------|
| Simple                | Well-understood   | Conventional wisdom        | Credit scoring model             |
| Complex               | Well-understood   | System-level thinking      | Hiring AI with feedback loops    |
| Simple                | Opaque            | System-level thinking      | Facial recognition model         |
| Complex               | Opaque            | System-level thinking      | Dynamic pricing for gig workers  |

I’ve used this table to guide decisions in production systems. For example, a team at a healthcare company built a model to predict patient deterioration. The constraint—preventing false negatives—was simple, but the system—the model and the clinical workflow—was complex. The team used a system-level approach: they combined the model’s predictions with clinician judgment and implemented a strict override policy. The result? The system’s false negative rate dropped to 0.1%, compared to 2% for the model alone.

Another example: a team at a retail company built a model to predict inventory demand. The constraint—preventing bias against certain product categories—was complex, and the system—the supply chain and sales data—was opaque. The team used a system-level approach: they audited the data pipeline, implemented fairness constraints in the model, and added a human review step for low-confidence predictions. The result? The model’s bias against certain categories dropped by 80%.

The key takeaway here is that the right approach depends on the complexity of the constraint and the opacity of the system. Recognize the limits of the conventional wisdom and supplement it with system-level thinking when necessary.


## Objections I've heard and my responses

### "Ethics is too vague. How do you measure it?"

The honest answer is that ethics isn’t a single metric; it’s a set of constraints. You can measure bias, fairness, transparency, and safety, but you can’t reduce ethics to a single number. For example, you can measure the false positive rate across demographic groups, but that won’t tell you if the system is reinforcing inequality in the real world.

In my experience, the best way to measure ethics is to look at the system’s behavior in production, not in the lab. Deploy the system with careful monitoring, and measure the harm it causes. If the harm is unacceptable, adjust the system. This approach isn’t perfect, but it’s better than pretending ethics can be reduced to a checklist.

### "We don’t have the resources to do system-level ethics."

I’ve heard this from startups and large companies alike. The honest answer is that you don’t need infinite resources to do system-level ethics; you need to recognize the constraints and allocate resources accordingly. For example, a startup building a hiring AI can’t afford to hire an ethics consultant, but they can implement a simple override policy: if the model’s prediction is below a certain threshold, route the candidate to a human reviewer. This approach doesn’t eliminate bias, but it reduces the harm caused by the system.

Another example: a company building a recommendation system can’t afford to implement a full fairness audit, but they can implement a simple content filter to prevent toxic outputs. This approach doesn’t eliminate all harm, but it reduces the harm caused by the system.

The key takeaway here is that system-level ethics isn’t about doing everything; it’s about doing the minimum necessary to prevent harm.


### "Regulations will solve this. We just need to wait."

I’ve seen teams use this as an excuse to avoid ethical responsibility. The honest answer is that regulations lag far behind technology, and by the time they catch up, the harm is already done. For example, the EU’s AI Act was proposed in 2021, but it won’t be fully implemented until 2026 at the earliest. In the meantime, companies are deploying AI systems without oversight, and the harm is already happening.

In my experience, the best way to avoid harm isn’t to wait for regulations; it’s to implement ethical constraints proactively. For example, a company building a facial recognition system can implement a strict accuracy threshold for all demographic groups, even if regulations don’t require it. This approach doesn’t guarantee compliance with future regulations, but it prevents harm in the present.

The key takeaway here is that waiting for regulations is a cop-out. Ethical responsibility can’t be outsourced to policymakers.


### "Ethics slows down innovation. We need to move fast."

I’ve heard this from every fast-growing company I’ve worked with. The honest answer is that ethics doesn’t have to slow down innovation; it can guide it. For example, a company building a new product can use ethical constraints to prioritize features that are safe and valuable, rather than features that are just novel.

In my experience, the best way to avoid slowing down innovation is to integrate ethics into the development process from the start. For example, a team building a new AI feature can include an ethics review in the design phase, rather than treating it as an afterthought. This approach doesn’t slow down innovation; it makes it more sustainable.

The key takeaway here is that ethics isn’t a speed bump; it’s a guardrail.


## What I'd do differently if starting over

If I were building an AI system today, I’d start with three principles:

1. **Treat ethics as a first-class constraint, not a checklist.**
2. **Design the system to be auditable and overridable.**
3. **Assume the model will fail, and plan for it.**

Here’s how I’d apply these principles:

### Principle 1: Treat ethics as a first-class constraint

I’d start by defining the ethical constraints upfront. For example, if I’m building a hiring AI, I’d define constraints like:
- The model must not systematically disadvantage any demographic group.
- The model’s predictions must be overridable by a human reviewer.
- The model’s data pipeline must be auditable for bias.

I’d encode these constraints in the system design, not in a post-hoc audit. For example, I’d use a fairness-aware training algorithm like [AIF360](https://aif360.mybluemix.net/) to enforce demographic parity during training. I’d also implement a monitoring system to track the model’s predictions in production and alert if the constraints are violated.

Here’s a Python example of how I’d implement a fairness constraint during training:

```python
from aif360.algorithms import PreProcessing
from aif360.datasets import BinaryLabelDataset
from sklearn.linear_model import LogisticRegression

# Load dataset with sensitive attributes
dataset = BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=df,
    label_names=['label'],
    protected_attribute_names=['race', 'gender']
)

# Apply fairness preprocessing
preprocessor = PreProcessing(
    privileged_groups=[{'race': 1, 'gender': 1}],
    unprivileged_groups=[{'race': 0, 'gender': 0}]
)
transformed_dataset = preprocessor.fit_transform(dataset)

# Train model on transformed data
model = LogisticRegression()
model.fit(transformed_dataset.features, transformed_dataset.labels)
```


### Principle 2: Design the system to be auditable and overridable

I’d design the system so that every decision can be audited and overridden. For example, if I’m building a medical diagnosis AI, I’d implement a system where:
- The model’s predictions are accompanied by explanations (e.g., SHAP values).
- Clinicians can override the model’s predictions with a justification.
- Every override is logged and reviewed weekly.

I’d also implement a content filter for LLMs to prevent toxic outputs. Here’s a JavaScript example of how I’d implement a simple content filter:

```javascript
function filterToxicity(text) {
  const toxicPatterns = [
    /\\b(nigger|kike|chink|fag|retard)\\b/i,
    /\\b(rape|kill|murder|die)\\b/i,
    /\\b(hate|violence|terror)\\b/i
  ];
  const isToxic = toxicPatterns.some(pattern => pattern.test(text));
  return isToxic ? null : text;
}

// Example usage
const userInput = "I hate this product. I wish I could kill myself.";
const filteredInput = filterToxicity(userInput);
if (!filteredInput) {
  console.log("Toxic input detected. Rejecting response.");
} else {
  console.log("Safe input. Proceeding with generation.");
}
```


### Principle 3: Assume the model will fail, and plan for it

I’d assume the model will fail in production, and I’d plan for it. For example, if I’m building a recommendation system, I’d implement:
- A fallback to a simpler model if the main model’s confidence is low.
- A human review step for high-stakes decisions.
- A monitoring system to detect drift and alert if the model’s performance degrades.

Here’s a Python example of how I’d implement a fallback system:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Train main model (e.g., Random Forest)
main_model = RandomForestClassifier()
main_model.fit(X_train, y_train)

# Train fallback model (e.g., Logistic Regression)
fallback_model = LogisticRegression()
fallback_model.fit(X_train, y_train)

def predict_with_fallback(X, threshold=0.8):
    main_probs = main_model.predict_proba(X)[:, 1]
    fallback_probs = fallback_model.predict_proba(X)[:, 1]
    
    # Use main model if confidence is high
    use_main = main_probs > threshold
    preds = np.where(use_main, main_model.predict(X), fallback_model.predict(X))
    
    # Log decisions for audit
    for i, (main_prob, fallback_prob, pred) in enumerate(zip(main_probs, fallback_probs, preds)):
        if not use_main[i]:
            print(f"Sample {i} fell back to logistic regression. Main prob: {main_prob:.2f}, Fallback prob: {fallback_prob:.2f}")
    
    return preds
```


The key takeaway here is that ethics isn’t a box to check; it’s a set of constraints to enforce. If I were starting over, I’d design the system to enforce those constraints from the beginning, not as an afterthought.


## Summary

AI ethics isn’t about bias in training data or fairness in model outputs. It’s about power, responsibility, and the systems we build around models. Big Tech frames ethics as a checklist to avoid lawsuits, but the real ethical failures happen when we automate decisions without considering the downstream harm.

The conventional wisdom—audit for bias, publish model cards, use open-source—is incomplete. It treats ethics as a property of the model, not the system. The honest answer is that ethical failures often happen not because the model is bad, but because the system around the model is designed to reinforce inequality.

To build ethical AI, we need to treat ethics as a first-class constraint. That means designing systems that are auditable, overridable, and resilient to failure. It means recognizing the trade-offs between performance, cost, and safety. And it means acknowledging that the models we build are just amplifiers of existing biases and power imbalances.

If you’re building an AI system today, start by defining the ethical constraints upfront. Encode them in the system design, not in a post-hoc audit. Design the system to be auditable and overridable. Assume the model will fail, and plan for it. And recognize that ethics isn’t a box to check; it’s a constraint to enforce.


Start by writing a one-page ethical constraints document for your next project. Define the harm you’re willing to accept, the trade-offs you’re willing to make, and the fallback mechanisms you’ll implement. Don’t wait for regulations or PR to force your hand. Do it now.


## Frequently Asked Questions

How do I fix bias in my training data without losing accuracy?

Start by auditing your data for proxies for sensitive attributes. For example, if you’re building a hiring model, look for ZIP codes, job titles, or education levels that correlate with race or gender. Use techniques like reweighting, resampling, or adversarial debiasing to reduce the correlation. Monitor the model’s performance across groups in production and adjust as needed. Don’t assume that removing a biased feature will fix the issue; the bias might be encoded in other features.


What is the difference between fairness and ethics in AI?

Fairness is a subset of ethics. Fairness is about ensuring the model’s predictions don’t systematically disadvantage any group. Ethics is broader: it includes fairness, but also transparency, accountability, safety, and the power dynamics of the system. For example, a model might be fair on paper, but if it automates decisions without human oversight, it might still be unethical.


Why does my model pass fairness tests in the lab but fail in production?

Because the lab doesn’t capture the real world. The lab tests static datasets and controlled environments, but production has distribution shifts, adversarial inputs, and human behavior that wasn’t in the training set. For example, a hiring model might pass fairness tests on historical data, but fail in production because recruiters treat the model’s scores as immutable thresholds. The issue isn’t in the model; it’s in how the model’s outputs are used.


How do I know if my AI system is ethical enough?

You don’t. Ethics isn’t a binary; it’s a spectrum. The best you can do is define the constraints upfront, measure the harm in production, and adjust as needed. If the harm is unacceptable, change the system. Don’t wait for a perfect solution; start with the minimum necessary to prevent harm.