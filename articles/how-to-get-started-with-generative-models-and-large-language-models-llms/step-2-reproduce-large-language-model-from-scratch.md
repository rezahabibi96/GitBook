# Step 2: Reproduce Large Language Model (from scratch)

## Let’s Continue with the Second Step

At this point, you've built a strong foundation in generative deep learning. Now it’s time to dive into how models like ChatGPT are actually built.

## Table of Contents

* [Let’s Continue with the Second Step](step-2-reproduce-large-language-model-from-scratch.md#lets-continue-with-the-second-step)
* [Table of Contents](step-2-reproduce-large-language-model-from-scratch.md#table-of-contents)
* [Build a Large Language Model (From Scratch)](step-2-reproduce-large-language-model-from-scratch.md#build-a-large-language-model-from-scratch)
  * [You'll Learn How To](step-2-reproduce-large-language-model-from-scratch.md#youll-learn-how-to)
  * [Outline of the Book](step-2-reproduce-large-language-model-from-scratch.md#outline-of-the-book)
  * [What You'll Achieve](step-2-reproduce-large-language-model-from-scratch.md#what-youll-achieve)
* [Want to Go Deeper?](step-2-reproduce-large-language-model-from-scratch.md#want-to-go-deeper)
* [Very Useful Alternative Resource](step-2-reproduce-large-language-model-from-scratch.md#very-useful-alternative-resource)

## <i class="fa-book-open">:book-open:</i> **Build a Large Language Model (From Scratch)**

This book is an excellent next step. Written by Sebastian Raschka, it guides you through the full process of developing a GPT-like Large Language Model (LLM) from scratch.

### **You'll Learn How To**

* Understand the architecture of LLMs
* Implement your own transformer model
* Pretrain it on a text corpus
* Finetune it for specific tasks

The book is highly practical and comes with a comprehensive [GitHub repository](https://github.com/rasbt/LLMs-from-scratch) full of hands-on examples built in PyTorch. Don’t worry if you’re new to PyTorch, the book even includes an appendix to help you get started with it.

### **Outline of the Book**

* Ch 1: Understanding Large Language Models
* Ch 2: Working with Text Data
* Ch 3: Coding Attention Mechanisms
* Ch 4: Implementing a GPT Model from Scratch
* Ch 5: Pretraining on Unlabeled Data
* Ch 6: Finetuning for Text Classification
* Ch 7: Finetuning to Follow Instructions

Unlike the first book where you could skip around, the chapters in this book are sequentially dependent, each one builds directly on the previous. So, it's best to read the book from Chapter 1 all the way through to Chapter 7, without skipping.

### What You'll Achieve

Once you finish this book, you'll not only understand how LLMs like ChatGPT work, but you'll also have built your own simplified version.&#x20;

And with that, you’re ready for Step 3: Reinforcement Learning from Human Feedback (RLHF), which is important for LLM reasoning.

## Want to Go Deeper?

This book focuses primarily on **practical implementation**. If you’re also interested in the **mathematical theory** behind LLMs (which not everyone is), there’s a helpful preprint on [arXiv: _Foundations of Large Language Models_](https://arxiv.org/abs/2501.09223) that complements the material.

## Very Useful Alternative Resource

If you prefer video over textbooks, one of the best resources for learning about LLMs comes from one of the most prominent researchers in the field, Andrej Karpathy. His video series is a must-watch due to the sheer depth and clarity of the explanations. Not only does he teach in an easy-to-understand manner, but he also codes everything step by step.

* [Deep Dive into LLMs like ChatGPT](https://youtu.be/7xTGNNLPyMI?si=b_ggnnBfeiKSJ9gN)\
  In this video, Karpathy introduces Large Language Models (LLMs), explains how they work, and covers some of the most recent updates. This is the first video you should watch to get a high-level understanding of the world of LLMs.
* [Let's build GPT: from scratch, in code, spelled out](https://youtu.be/kCc8FmEb1nY?si=cJVTznDaJ5rm34LO)\
  Here, he focuses on building GPT, specifically GPT-2, from scratch, walking through the entire process in code.
* [Let's build the GPT Tokenizer](https://youtu.be/zduSFxRajkE?si=wpxTggVPN5gV7ifJ)\
  This video explains how **tokenization** works, the process of converting text into tensors or vectors so it can be processed by a model. Karpathy builds a tokenizer from scratch using **Byte Pair Encoding (BPE)**.
* [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU?si=mWxSDEn6xq5BT0tL)\
  This is an advanced follow-up to the earlier videos. It not only builds GPT-2 but also reproduces its results using real-world optimization and training strategies.

