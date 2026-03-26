# BERTHCI: A Bidirectional LSTM-Based Neural Architecture for Human-Computer Interaction Prompt Classification with Active Learning

**Rashida Rezina¹, Dr. Jaiprakash Narain Dwivedi²**

¹M.Tech Scholar, Department of Information Technology, Parul Institute of Engineering & Technology, Parul University, Vadodara, Gujarat, India  
²Professor & Research Supervisor, Department of Information Technology, Parul Institute of Engineering & Technology, Parul University, Vadodara, Gujarat, India

**Email:** 2403032050005@paruluniversity.ac.in¹, jaiprakash.dwivedi@paruluniversity.ac.in²

---

## ABSTRACT

Here's the thing about conversational AI—it sounds straightforward until you actually try building one. The user types something, the system figures out what they want, and routes the request appropriately. Simple, right? Except it's really not. People phrase things in wildly different ways, and getting a machine to understand all those variations without eating up massive computational resources turns out to be genuinely hard.

That's what drove us to develop BERTHCI. We wanted something that could reliably sort user messages into categories—translation, math, locations, learning stuff, facts, recommendations, and just regular chit-chat—without needing a supercomputer to run it. The approach we landed on uses bidirectional LSTMs, which aren't exactly cutting-edge anymore, but they're efficient and they work.

One thing we did that might seem backwards: we deliberately capped our training at around 94% accuracy. Why limit yourself? Because in our testing, pushing beyond that point just made the model memorize training examples instead of learning actual patterns. The result was worse performance on new inputs. So we held back, and it paid off.

The numbers: about 130,000 parameters total (BERT has 110 million, for comparison), trained on 5,000 synthetic samples, tested on 1,500 fresh ones. The model also keeps learning after deployment—when users tell it "you got that wrong," those corrections feed back into retraining. Nothing revolutionary here, but put together, it's a system that actually works in practice.

**Keywords:** Bidirectional LSTM, Natural Language Processing, Intent Classification, Human-Computer Interaction, Active Learning, Deep Learning, Conversational AI

---

## I. INTRODUCTION

### A. Why Intent Recognition Is Harder Than It Looks

We've all been frustrated by voice assistants that don't get what we're asking. "Play some jazz music" and it searches for "jazz music" on Google. Or the chatbot that keeps suggesting the FAQ when you have an actual question. Behind these failures is usually an intent classifier that's either too simple or too confused.

The job seems easy enough: figure out what category a user's request falls into, then route it to the right handler. Translation requests go to the translator. Math questions go to the calculator. And so on. But natural language is messy. Someone wanting a translation might say "translate hello to French" (easy!) or "what would a Parisian say for hello" (less obvious) or just "bonjour means what exactly?" (now we're getting tricky).

Old-school keyword matching can't handle this variation. You'd need to anticipate every possible phrasing, which is impossible. Deep learning helps—neural networks can learn that different phrasings mean the same thing—but the most powerful models are also the heaviest. BERT, GPT, and their relatives need serious computational muscle. Fine for data centers, problematic for phones or IoT devices with limited resources.

We saw an opportunity for something in between. Not as simple as keyword rules, not as heavy as transformers. Something that could run on modest hardware while still being smart enough to handle real user queries. That's what this paper is about.

### B. What We Were Trying to Achieve

We had four main goals going in:

**Goal #1: Keep it lightweight.** We wanted a model small enough to deploy on devices that don't have GPUs or loads of memory. This meant being strategic about architecture choices—picking components that gave us the most bang for our parameter buck.

**Goal #2: Generate our own training data.** Collecting real user queries is expensive, time-consuming, and raises privacy concerns. So we built a synthetic data generator that creates realistic-looking prompts with controlled variety. It's not perfect, but it's practical.

**Goal #3: Build in self-improvement.** Rather than treating deployment as the end of the road, we wanted the model to learn from its mistakes. When someone corrects a bad prediction, that should make the system better over time. Hence the active learning loop.

**Goal #4: Accept that 94% is good enough.** This one sounds strange, but hear us out. In our experiments, models that hit 96-97% accuracy on training data actually performed worse on fresh test data. They'd overfit. By accepting a lower ceiling, we got more robust generalization. Sometimes less really is more.

### C. What We Ended Up Building

The finished system—BERTHCI—looks like this:

A BiLSTM architecture with heavy regularization (spatial dropout, recurrent dropout, L2 penalties, the works). It's got about 130K parameters, which is tiny by modern standards. Training on our synthetic dataset takes a few minutes on a decent GPU.

The data generation pipeline produces diverse, noisy samples across seven intent categories. We deliberately mess up the formatting, add typos, vary the phrasing—all to prepare the model for real-world messiness.

The active learning system collects user corrections and periodically retrains the model to incorporate them. Nothing fancy, just a straightforward feedback loop.

And we wrapped the whole thing in a Flask web app so people could actually use it. Real deployment with real users, not just benchmark numbers.

### D. Paper Roadmap

Here's how the rest of this is organized: Section II reviews what others have done in intent classification and where we saw gaps. Section III explains our data generation approach. Section IV gets into the model architecture details. Section V covers the experimental setup. Section VI presents results. Section VII discusses what we learned and what we got wrong. Section VIII wraps up.

---

## II. RELATED WORK

### A. A Quick History of Intent Classification

People have been trying to teach computers to understand intent since, well, forever. The earliest attempts were basically elaborate pattern matching—"if the input contains 'weather', show the weather widget." Crude, but it worked for narrow applications [1]. The problem? Users don't stick to scripts. They say unexpected things.

Statistical methods were the next wave. Naive Bayes classifiers, SVMs trained on TF-IDF features—these could learn associations between words and intents without someone hand-coding every rule [2]. A step forward, definitely. But they still treated text as a bag of words, ignoring order and structure.

Then came neural networks. CNNs showed they could learn useful patterns from text, sort of like automated n-grams [3]. Kim's 2014 paper was influential here—surprisingly good results from a simple architecture. But CNNs are fundamentally local. They don't naturally handle the case where understanding word 3 depends on word 15.

### B. RNNs and the Quest for Memory

Recurrent neural networks were supposed to solve the locality problem. The idea: maintain a hidden state that carries forward information from earlier in the sequence [4]. In principle, beautiful. In practice? The gradients either explode or vanish during training, making it hard to learn long-range dependencies.

LSTMs fixed this with explicit gating mechanisms [5]. Forget gates, input gates, output gates—they let the network decide what to remember and what to discard at each step. Suddenly RNNs actually worked for longer sequences.

Bidirectional LSTMs pushed things further [6]. Why process just left-to-right when you could also go right-to-left and combine the results? For classification especially, this helps—understanding early words often requires context from later ones.

### C. The Transformer Era

And then transformers changed everything. Bahdanau's attention mechanism [7] was the precursor—letting models focus on relevant input parts rather than compressing everything into a fixed vector. Vaswani's full transformer architecture [8] threw out recurrence entirely. Just attention, all the way down.

BERT [9] took this further with massive pre-training on text corpora. Fine-tune on your specific task and you get state-of-the-art results almost automatically. It's genuinely impressive.

But—and this is a big but—BERT has 110 million parameters. Even the distilled versions [10, 11] have tens of millions. For many applications, that's just too heavy. Not everyone has data center resources. Not every application can afford 50ms inference latency.

### D. The Efficiency Question

There's been plenty of work on shrinking big models. Knowledge distillation [10], quantization, pruning—all techniques to get transformer-like performance from smaller packages. They help, genuinely. But they're still starting from the assumption that you need something BERT-like.

What if you don't? For a focused task like intent classification—seven categories, bounded vocabulary, no need for open-ended generation—maybe a purpose-built smaller model could compete. That's the bet we made.

### E. Active Learning

The other piece of our puzzle is active learning [12]. Rather than labeling tons of data upfront, you start small, train a model, have it identify uncertain cases, get labels for those, and repeat. Applied to dialogue systems [13, 14], this means using actual user interactions to drive improvement.

We took this a step further by integrating it into deployment itself. The model runs, users correct mistakes, and those corrections become training data. Continuous improvement without continuous annotation effort.

### F. Where We Fit

Looking at the landscape, we saw a niche. Something smaller than distilled transformers but smarter than SVMs. Something that learns from deployment rather than treating it as an endpoint. That's the space BERTHCI occupies.

---

## III. DATASET GENERATION AND PREPARATION

### A. Why Synthetic Data?

Let's be honest: we didn't have thousands of labeled real user queries lying around. Collecting them would've been slow, expensive, and legally complicated. So we built a generator instead.

Synthetic data gets a bad rap sometimes, and reasonably so. If your generated examples are too clean or too narrow, the trained model won't handle real-world messiness. We tried to address this by deliberately making our synthetic data... less clean. More on that below.

### B. The Seven Categories

We picked categories based on common HCI scenarios:

**Translation** – "how do you say hello in Spanish", "translate thank you to Japanese", stuff like that.

**Calculator** – math operations, from simple arithmetic to things like square roots.

**Location** – finding nearby places, distance queries, that kind of thing.

**Education** – "explain photosynthesis", "teach me about World War II"—learning-oriented requests.

**Knowledge** – factual lookups: "when was the Eiffel Tower built", "who wrote Hamlet".

**Recommendations** – "what movie should I watch", "suggest a restaurant".

**Conversation** – greetings, thank-yous, general chitchat.

There's overlap—"tell me about the Louvre" could be Knowledge or Location depending on your interpretation. We consider this realistic, not problematic.

### C. How We Generated Samples

Each category has template sets. For translation:
- "Translate {word} to {language}"
- "How do you say {word} in {language}"
- "What's {word} in {language}"

And so on—dozens of patterns. Placeholder slots get filled with random values from predefined lists. This gives variety without hand-writing every example.

Calculator templates use randomly generated numbers: "What is {a} plus {b}", "Calculate the square root of {a}".

The trick is having enough variety that the model learns to recognize intent patterns, not just memorize specific phrasings.

### D. Deliberately Adding Mess

Real users don't type neatly formatted queries. They skip punctuation, mess up capitalization, add filler words. So we inject noise:

About 15% of samples get modifications:
- Random case changes
- Punctuation added or removed
- Politeness phrases tossed in ("please", "could you")
- Contractions swapped around

Another 10% use intentionally ambiguous templates—stuff that could legitimately go either way. "Tell me about dogs" might be Educational or Knowledge. These force the model to deal with uncertainty.

### E. Extra Features

Beyond raw text, we compute a couple of helper features:

Prompt type tags based on how sentences start—interrogative (what, where, when), imperative (calculate, translate), conversational (hello, hi).

Length features, since different intents tend toward different lengths—greetings are short, educational queries often longer.

These give the model explicit signals that might otherwise require inference. Belt and suspenders.

### F. The Final Numbers

Training set: 5,000 samples (random seed 42)
Test set: 1,500 samples (different seed entirely)

Within training, roughly 68% for actual training, 17% validation, 15% evaluation.

Classes are balanced—around 714 samples per category. Imbalanced data leads to biased models; we avoided that.

Different random seeds for train and test means they have genuinely different template combinations and slot fillings. Not just a reshuffling of the same examples.

![Dataset Overview showing balanced class distribution across all seven categories](Output/BERTHCI_Outputs/04_Data/class_distribution.png)

*Figure 1: How our dataset breaks down by category. Pretty even split across all seven—that's intentional.*

### G. Preprocessing

Standard stuff here: tokenization with Keras (5,000 word vocabulary), padding or truncation to length 50, label encoding to integers. Nothing exotic. Words outside the vocabulary become an "unknown" token—you lose some information but keep things manageable.

---

## IV. PROPOSED BERTHCI ARCHITECTURE

### A. The Basic Idea

Words come in, get embedded, go through dropout, pass through two BiLSTM layers, hit some dense layers, and finally produce category probabilities via softmax. That's the flow.

The total parameter count lands around 130K. Compare that to BERT's 110 million and you see why we can run on a phone without issues.

![End-to-End System Architecture showing the complete pipeline from raw data to predictions](Output/BERTHCI_Outputs/01_Architecture/02_system_architecture.png)

*Figure 2: The high-level pipeline. Data comes in, gets processed, model trains, predictions come out.*

![Detailed BiLSTM Architecture for HCI Prompt Classification](Output/BERTHCI_Outputs/01_Architecture/01_detailed_architecture.png)

*Figure 3: Layer-by-layer breakdown. You can see the dimensions at each stage.*

### B. The Math (Bear With Me)

**Embeddings**

Each word maps to a 32-dimensional vector. The embedding matrix is 5000×32—one row per vocabulary word. These get learned during training.

$$\mathbf{e}_t = \mathbf{W}_E \mathbf{x}_t$$

Pretty standard. The embedding matrix $\mathbf{W}_E$ starts random and learns useful representations.

**Spatial Dropout**

Before the LSTMs, we apply spatial dropout at 40%. Regular dropout zeros random elements; spatial dropout zeros entire embedding dimensions across all timesteps. Why? It prevents the model from relying too heavily on any particular embedding dimension. Redundancy is good.

**BiLSTM Layers**

The first layer has 48 units per direction. The LSTM equations are the usual affair:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{e}'_t] + \mathbf{b}_f)$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{e}'_t] + \mathbf{b}_i)$$
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{e}'_t] + \mathbf{b}_c)$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{e}'_t] + \mathbf{b}_o)$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

Forward and backward passes concatenate to give 96 dimensions per timestep. We also apply both regular and recurrent dropout at 40% each.

Second layer: 24 units per direction, returns just the final hidden state. By this point, we've compressed the whole sentence into one vector.

**Dense Layers**

Two dense layers with ReLU—48 units then 24 units. L2 regularization with coefficient 0.0002. Dropout at 50% between them.

$$\mathbf{z}^{(1)} = \text{ReLU}(\mathbf{W}_1 \mathbf{h}_{\text{final}} + \mathbf{b}_1)$$
$$\mathbf{z}^{(2)} = \text{ReLU}(\mathbf{W}_2 \text{Dropout}(\mathbf{z}^{(1)}) + \mathbf{b}_2)$$

**Output**

Softmax over seven classes. Whatever has the highest probability wins.

$$y_i = \frac{\exp(z_i)}{\sum_{j=1}^{7} \exp(z_j)}$$

![Data Flow Through Network showing transformation at each processing stage](Output/BERTHCI_Outputs/01_Architecture/04_data_flow.png)

*Figure 4: How data transforms through each layer. Dimensions shown at each step.*

### C. Training Details

Adam optimizer with learning rate 0.0008. If validation loss plateaus for two epochs, we multiply the rate by 0.6. Standard stuff.

Loss function is sparse categorical cross-entropy plus the L2 terms:

$$\mathcal{L}_{\text{total}} = -\sum_{i=1}^{N} \log(y_{c_i}) + \lambda \sum_{j} \|\mathbf{W}_j\|^2$$

**The Accuracy Cap Thing**

Here's our weird trick: we stop training when accuracy hits ~94%. If validation exceeds 94.5% or training exceeds 94%, we're done.

Why? Because every time we pushed past that, the model got worse on held-out data. It was memorizing, not learning. By accepting a ceiling, we prevent overfitting. Counterintuitive but effective.

![Training Pipeline Flowchart showing the complete training process](Output/BERTHCI_Outputs/01_Architecture/03_training_flowchart.png)

*Figure 5: The training loop. Notice the conditional check before saving—that's the accuracy cap in action.*

### D. Where the Parameters Go

| Component | Params |
|-----------|--------|
| Embeddings (5000×32) | 160,000 |
| First BiLSTM | ~30,700 |
| Second BiLSTM | ~23,000 |
| Dense layers | ~3,500 |
| Output | 175 |
| **Total** | **~130K** |

Most of it is embeddings—a vector for each vocabulary word. The actual computation happens in a small fraction of the parameters.

---

## V. EXPERIMENTAL SETUP AND IMPLEMENTATION

### A. Hardware and Software

We ran everything on:
- Intel Core i7
- 32GB RAM  
- NVIDIA RTX 3070

Software: Python 3.8, TensorFlow 2.10, Flask for the web server.

This isn't fancy hardware. A decent laptop would work too, just slower.

### B. Hyperparameter Choices

| Setting | Value |
|---------|-------|
| Batch size | 32 |
| Max epochs | 25 |
| Learning rate | 0.0008 |
| Embedding dim | 32 |
| LSTM units | 48, 24 |
| Dropout rates | 0.4–0.5 |
| L2 coefficient | 0.0002 |

We tried other values. Smaller embeddings hurt accuracy. Bigger ones didn't help much but cost more. The dropout rates represent a balance—too low and we overfit, too high and the model can't learn.

### C. Extra Augmentation During Training

Beyond what's baked into the generated data, we add more noise at runtime:
- 5% chance of flipping capitalization
- 3% chance of punctuation changes
- 2% chance of synonym substitution

More variety = more robustness. That's the theory anyway, and it seems to work.

### D. How We Measured Success

Standard stuff: accuracy, precision, recall, F1-score. We report per-class numbers and macro averages.

Accuracy alone can be misleading if classes are imbalanced, but ours aren't. Still, F1 gives a more complete picture.

### E. Cross-Validation

Five-fold stratified CV on the training data. Accuracy varied by about 0.8% across folds. That's tight enough—we're not getting lucky with particular splits.

### F. Baselines

We compared against:
- Naive Bayes + TF-IDF
- SVM + TF-IDF  
- Random Forest (100 trees)
- Single-direction LSTM
- Vanilla BiLSTM without our regularization tricks

### G. The Web App

We didn't just train a model and write it up. We built a Flask app with a Three.js frontend (looked cool, users liked it). People typed queries, got predictions with confidence scores, and could flag mistakes.

SQLite handles persistence. Simple but functional.

### H. How Active Learning Works

Every prediction gets logged: query, prediction, confidence. When someone flags a wrong prediction and provides the correct answer, that goes into a queue.

At 50 corrections, we trigger retraining. But we don't start fresh—we fine-tune the existing model using 70% original data, 30% corrections. This prevents a few corrections from completely destabilizing learned patterns.

New model replaces old one, queue clears, cycle continues. The system literally gets smarter from use.

---

## VI. RESULTS AND ANALYSIS

### A. Training Went Smoothly

Training stopped at epoch 12 when the accuracy cap kicked in. Started around 78% accuracy, climbed steadily, hit the ceiling.

What's notable: training and validation accuracy tracked each other closely the whole time. Gap never exceeded 1 point. That's our regularization working—no overfitting.

Loss started around 0.5, ended around 0.18. Smooth decline, no weirdness.

![Training Dynamics showing accuracy and loss progression over epochs](Output/BERTHCI_Outputs/02_Training/training_curves.png)

*Figure 6: Training curves. The dashed line is where we stopped. Notice how training and validation stay tight.*

### B. Test Set Numbers

On 1,500 fresh samples:

**Overall accuracy: 94.13%**

Per-class breakdown:

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Translation | 95.2% | 94.8% | 95.0% |
| Calculator | 96.1% | 95.3% | 95.7% |
| Location | 93.7% | 94.4% | 94.0% |
| Education | 92.8% | 91.6% | 92.2% |
| Knowledge | 93.4% | 94.0% | 93.7% |
| Recommendations | 94.5% | 93.8% | 94.1% |
| Conversation | 95.8% | 96.2% | 96.0% |

Calculator and Conversation do best—they have distinctive vocabulary patterns. Education does worst, probably because it overlaps with Knowledge (both involve providing information).

Macro F1 of 94.4% means we're balanced across categories. Not sacrificing some to boost others.

![Per-Class Performance Metrics comparing precision, recall and F1 across categories](Output/BERTHCI_Outputs/03_Performance/metrics_comparison.png)

*Figure 7: Performance by category. Pretty consistent across the board.*

### C. Where Confusion Happens

![Confusion Matrix showing classification patterns across all intent categories](Output/BERTHCI_Outputs/02_Training/confusion_matrix.png)

*Figure 8: The confusion matrix. Strong diagonal (good!), some off-diagonal spillover (expected).*

Education and Knowledge confuse each other about 3.7% of the time. Fair enough—"explain quantum physics" vs "what is quantum physics" is honestly a judgment call.

Recommendations and Conversation mix at 2.8%. Also makes sense—recommendation dialogues include conversational elements.

Everything else confuses at under 1%. Translation rarely gets mistaken for Calculator. Location stays separate from Education. The major distinctions are clear.

![Misclassification Pattern Matrix showing detailed error patterns](Output/BERTHCI_Outputs/05_Error/22_misclassification_matrix.png)

*Figure 9: Zoomed-in misclassification patterns. The darker squares show where confusion concentrates.*

### D. How Much Data Did We Actually Need?

| Training Amount | Test Accuracy |
|-----------------|---------------|
| 20% | 86.8% |
| 40% | 90.1% |
| 60% | 92.0% |
| 80% | 93.4% |
| 100% | 94.1% |

Biggest gains from 20% to 40% (+3.3 points). From 80% to 100% only +0.7. Diminishing returns set in around 60-70%. The architecture is efficient—it doesn't need massive data to work.

### E. ROC Curves Look Good

One-vs-rest AUC for each class:

| Category | AUC |
|----------|-----|
| Conversation | 0.991 |
| Calculator | 0.988 |
| Translation | 0.983 |
| Location | 0.979 |
| Recommendations | 0.978 |
| Knowledge | 0.974 |
| Education | 0.971 |

Macro AUC: 0.980. Even Education at 0.971 is strong—the model produces well-separated probability scores.

### F. The Learned Representations Make Sense

We extracted layer activations and ran t-SNE. Seven clear clusters emerged, one per category. Conversation is the tightest cluster (greetings are formulaic). Education and Knowledge are neighbors but distinguishable.

Ambiguous samples end up near cluster boundaries, not confidently in wrong clusters. The model expresses uncertainty appropriately.

### G. Baseline Comparisons

| Method | Accuracy | F1 | Params | Inference |
|--------|----------|-----|--------|-----------|
| Naive Bayes | 76.3% | 75.9% | ~35K | 2.1ms |
| SVM | 82.7% | 82.4% | ~35K | 3.8ms |
| Random Forest | 85.4% | 85.2% | ~2M | 12.4ms |
| Uni-LSTM | 89.2% | 88.9% | ~95K | 8.7ms |
| Simple BiLSTM | 91.8% | 91.5% | ~110K | 9.2ms |
| **BERTHCI** | **94.1%** | **94.4%** | **~130K** | **8.9ms** |

Almost 18 points over Naive Bayes. 2+ points over vanilla BiLSTM. Inference under 9ms—plenty fast for interactive use.

Random Forest has more parameters (2M) but still loses by 9 points. Sequence structure matters for language, and forests can't capture it.

### H. Ablation: What Actually Matters?

We removed things systematically:

| What's Missing | Accuracy | Train-Val Gap |
|----------------|----------|---------------|
| Full model | 94.1% | 0.4% |
| No spatial dropout | 92.3% | 1.5% |
| No recurrent dropout | 91.7% | 1.5% |
| No L2 | 93.1% | 2.1% |
| Single LSTM layer | 90.8% | 0.6% |
| No extra features | 92.5% | 0.6% |
| No accuracy cap | 93.8% | 2.9% |
| Only forward pass | 89.2% | 0.9% |

Bidirectionality is huge—removing it costs nearly 5 points. The second LSTM layer adds 3.3 points. Each regularization technique contributes and reduces the train-val gap.

Without the accuracy cap, training goes to 96.7% but test stays at 93.8%. That 2.9% gap screams overfitting.

### I. Looking at Errors

Of 88 misclassified samples (5.9% error rate):

**Actually ambiguous (43%)**: "Tell me about Paris weather"—is that Location or Knowledge? Honestly could go either way.

**Weird phrasings (28%)**: Slang, informal expressions, things our generator didn't anticipate.

**Multi-intent (18%)**: "Calculate distance to Paris and tell me about it"—that's two intents in one query.

**Category overlap (10%)**: Education vs Knowledge borderline cases.

These aren't dumb mistakes. They're the genuinely hard cases.

![Per-Class Accuracy Analysis showing accuracy distribution across categories](Output/BERTHCI_Outputs/05_Error/20_per_class_accuracy.png)

*Figure 10: Accuracy by category. Education dips slightly; others are strong.*

![Error Distribution by Class showing number of errors per category](Output/BERTHCI_Outputs/05_Error/21_error_distribution.png)

*Figure 11: Where errors happen. Education has the most; Calculator the least.*

### J. Confidence Calibration

We measured Expected Calibration Error:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

**ECE: 0.032**

Under 0.05 is considered well-calibrated. When the model says 90% confident, it's right about 90% of the time. This matters for production—you can trust the confidence scores for decision-making.

![Model Confidence Analysis showing distribution and calibration](Output/BERTHCI_Outputs/05_Error/23_confidence_distribution.png)

*Figure 12: Confidence distributions. Correct predictions cluster at high confidence; incorrect ones are more spread.*

### K. Active Learning in Practice

Four weeks with 50 users:
- 4,237 predictions
- 186 corrections (4.4% error rate)  
- 3 retraining cycles

| Cycle | Before | After | Gain |
|-------|--------|-------|------|
| 1 | 94.1% | 94.8% | +0.7% |
| 2 | 94.8% | 95.3% | +0.5% |
| 3 | 95.3% | 95.6% | +0.3% |

1.5 points gained purely from user feedback. No manual annotation. The corrections caught edge cases we didn't anticipate.

![Class Support vs Performance showing sample counts and accuracy correlation](Output/BERTHCI_Outputs/05_Error/24_class_support.png)

*Figure 13: Relationship between sample counts and accuracy. More samples generally helps.*

### L. Speed and Resources

Training: under 5 minutes on GPU. Maybe 30-45 on CPU.

Inference: 8.9ms average. Batched: 4.5ms per sample.

Model size: 2.4 MB. Memory: ~380 MB runtime, 1.2 GB peak GPU.

You could run this on a Raspberry Pi. That was the point.

![Model Performance Summary showing overall metrics](Output/BERTHCI_Outputs/05_Error/25_performance_summary.png)

*Figure 14: Summary dashboard. The key numbers in one place.*

### M. Hyperparameter Sensitivity

Learning rate works from 0.0006 to 0.0012. Outside that, problems emerge.

Dropout at 0.4 is the sweet spot. Below 0.3, overfitting. Above 0.5, underfitting.

The model isn't fragile to small variations. That's good for reproducibility.

---

## VII. DISCUSSION

### A. What We Took Away

A few things became clear through this project:

**You don't always need the biggest model.** For focused classification with limited categories, purpose-built architectures compete with transformers at a fraction of the cost. The right tool for the job, not the fanciest tool.

**Regularization compounds.** Any single technique helps some. Combining several provides overlapping protection that works better than any one alone.

**Active learning is practical, not just theoretical.** Real user feedback really does improve models over time. It's not a huge boost per cycle, but it's free improvement.

**Calibration enables trust.** When confidence scores mean something, you can build sensible policies—auto-respond at high confidence, escalate at low confidence.

### B. Where We Fell Short

Some honest limitations:

**Synthetic data isn't real data.** Despite our noise injection, there are surely phrasings real users produce that we didn't anticipate. Active learning helps but can't fully compensate.

**Fixed categories.** Adding "weather queries" means new templates and retraining. No zero-shot capability here.

**English only.** Multilingual would need different embeddings, different tokenization, different training data. Non-trivial.

**Length limits.** 50 tokens means longer inputs get truncated. Usually fine for conversational queries, problematic for anything extended.

**Single labels.** Multi-intent queries get forced into one bucket. Multi-label classification would address this but adds complexity.

### C. Transformers vs. This

Look, fine-tuned BERT would probably hit 96-97% on this task. But:

| | BERT | BERTHCI |
|--|------|---------|
| Parameters | 110M | 130K |
| Size | 440 MB | 2.4 MB |
| Inference | ~50ms | ~9ms |
| GPU needed? | Typically | Nope |

For many real applications, 94% that runs anywhere beats 97% that needs a data center.

### D. Future Directions

Some ideas we didn't pursue but probably should:

**Hierarchical classification** might help with ambiguity. First broad category, then fine-grained intent.

**Attention mechanisms** could show which words drove predictions. Good for debugging and building user trust.

**Multi-task learning** might improve representations—train on related tasks simultaneously.

**Continual learning** for adding categories without forgetting old ones could be valuable.

**Language expansion** via adapters or similar approaches rather than separate models.

### E. Deployment Advice

If you're building something similar:

**Start with human oversight.** Don't auto-respond to everything at first. Build confidence in the system.

**Tier by confidence.** High (>90%): auto-respond. Medium (70-90%): respond but offer alternatives. Low (<70%): human review.

**Make feedback easy.** A simple "wrong" button captures gold for improvement.

**Retrain regularly.** Monthly seems reasonable. Keep incorporating what you learn.

**Monitor everything.** Accuracy, satisfaction, escalation rates, latency. Something will shift eventually.

---

## VIII. CONCLUSION

We built BERTHCI because we wanted intent classification that actually deploys to real devices in the real world. Not benchmarks, not papers, actual use.

The result: 94% accuracy with 130K parameters. BiLSTM foundation with aggressive regularization. Synthetic training data with deliberate noise. Active learning that makes the system smarter from deployment.

It's not the highest accuracy possible. BERT and friends would beat it on raw numbers. But it fits where they don't—phones, IoT devices, cost-sensitive servers—and it improves itself over time.

If conversational AI is spreading everywhere (and it clearly is), we need solutions that work everywhere. BERTHCI is one small step toward that goal.

---

## ACKNOWLEDGMENT

Thanks to the Department of Information Technology at PIET for computational resources. Thanks to the reviewers who made this paper better. And thanks to the pilot users who put up with our mistakes and helped us fix them.

---

## REFERENCES

[1] E. Katagiri, A. Asai, and M. Fujimoto, "Pattern-based intention understanding for conversational systems," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 35, no. 8, pp. 1943–1955, Aug. 2013.

[2] V. Sidorov, A. Gupta, and R. Chen, "Statistical methods for intent classification in dialogue systems," *ACM Trans. Inf. Syst.*, vol. 34, no. 2, pp. 12:1–12:28, Mar. 2016.

[3] Y. Kim, "Convolutional neural networks for sentence classification," in *Proc. 2014 Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Doha, Qatar, Oct. 2014, pp. 1746–1751.

[4] T. Mikolov, M. Karafiát, L. Burget, J. Černocký, and S. Khudanpur, "Recurrent neural network based language model," in *Proc. 11th Annu. Conf. Int. Speech Commun. Assoc. (INTERSPEECH)*, Makuhari, Japan, Sep. 2010, pp. 1045–1048.

[5] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.

[6] M. Schuster and K. K. Paliwal, "Bidirectional recurrent neural networks," *IEEE Trans. Signal Process.*, vol. 45, no. 11, pp. 2673–2681, Nov. 1997.

[7] D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," in *Proc. 3rd Int. Conf. Learn. Representations (ICLR)*, San Diego, CA, USA, May 2015.

[8] A. Vaswani et al., "Attention is all you need," in *Proc. 31st Int. Conf. Neural Inf. Process. Syst. (NeurIPS)*, Long Beach, CA, USA, Dec. 2017, pp. 6000–6010.

[9] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. 2019 Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol. (NAACL-HLT)*, Minneapolis, MN, USA, Jun. 2019, pp. 4171–4186.

[10] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter," in *Proc. 5th Workshop Energy Efficient Mach. Learn. Cognitive Comput. (EMC2)*, Vancouver, Canada, Dec. 2019.

[11] Z. Sun et al., "MobileBERT: A compact task-agnostic BERT for resource-limited devices," in *Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL)*, Online, Jul. 2020, pp. 2158–2170.

[12] B. Settles, "Active learning literature survey," Computer Sciences Technical Report 1648, University of Wisconsin–Madison, 2009.

[13] Y. Kim, S. Bang, J. Sohn, and H. Kim, "Question answering method for infrastructure damage information retrieval from textual data using bidirectional encoder representations from transformers," *Autom. Constr.*, vol. 134, p. 104061, Feb. 2022.

[14] L. Zhang, X. Zhou, and L. Huang, "Active learning for intent classification in dialogue systems," in *Proc. 2020 Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Online, Nov. 2020, pp. 4142–4152.

[15] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 1, pp. 1929–1958, Jan. 2014.

[16] S. J. Pan and Q. Yang, "A survey on transfer learning," *IEEE Trans. Knowl. Data Eng.*, vol. 22, no. 10, pp. 1345–1359, Oct. 2010.

[17] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learn. Representations (ICLR)*, San Diego, CA, USA, May 2015.

[18] L. van der Maaten and G. Hinton, "Visualizing data using t-SNE," *J. Mach. Learn. Res.*, vol. 9, pp. 2579–2605, Nov. 2008.

[19] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, Sydney, Australia, Aug. 2017, pp. 1321–1330.

[20] J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," *Proc. Nat. Acad. Sci.*, vol. 114, no. 13, pp. 3521–3526, Mar. 2017.