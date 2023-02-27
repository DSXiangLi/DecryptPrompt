# DecryptPrompt

试(努)图(力)理解ChatGPT的超能力来源，顺手梳理下Prompt范式相关模型，持续更新中，有推荐paper和resources欢迎PR哟~~

每个方向推荐1-2篇五星好评的论文，没有推荐的方向就是我也还没看完哈哈哈~

## My blogs
- [解密Prompt系列1. Tunning-Free Prompt：GPT2 & GPT3 & LAMA & AutoPrompt](https://cloud.tencent.com/developer/article/2215545?areaSource=&traceId=)
- [解密Prompt系列2. 冻结Prompt微调LM： T5 & PET & LM-BFF](https://cloud.tencent.com/developer/article/2223355?areaSource=&traceId=)

## Papers
### Survey
- Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing :star2::star2::star2::star2::star2:
- Paradigm Shift in Natural Language Processing
- Pre-Trained Models: Past, Present and Future

### LLM Ability Analysis & Probing 
- How does in-context learning work? A framework for understanding the differences from traditional supervised learning
- Why can GPT learn in-context? Language Model Secretly Perform Gradient Descent as Meta-Optimizers
- Emerging Ability of Large Language Models
- Rethinking the Role of Demonstrations What Makes incontext learning work?
- Can Explanations Be Useful for Calibrating Black Box Models

### Tunning Free Prompt
- GPT2: Language Models are Unsupervised Multitask Learners
- GPT3: Language Models are Few-Shot Learners   :star2::star2::star2::star2::star2:
- LAMA: Language Models as Knowledge Bases?
- AutoPrompt: Eliciting Knowledge from Language Models

### Fix-Prompt LM Tunning
- T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- PET-TC(a): Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference  :star2::star2::star2::star2::star2:
- PET-TC(b): PETSGLUE It’s Not Just Size That Matters Small Language Models are also few-shot learners
- GenPET: Few-Shot Text Generation with Natural Language Instructions
- LM-BFF: Making Pre-trained Language Models Better Few-shot Learners
- ADEPT: Improving and Simplifying Pattern Exploiting Training

### Fix-LM Prompt Tunning 
- Prefix-tuning: Optimizing continuous prompts for generation  :star2::star2::star2::star2::star2:
- Prompt-tunning: The power of scale for parameter-efficient prompt tuning.
- WARP: Word-level Adversarial ReProgramming

### LM + Prompt Tunning 
- P-tunning: GPT Understands Too
- P-tunning v2: Prompt Tuning Can Be Comparable to Fine-tunning Universally Across Scales and Tasks
- PTR: Prompt Tuning with Rules for Text Classification
- PADA: Example-based Prompt Learning for on-the-fly Adaptation to Unseen Domains

### Instruction Tunning LLMs 
- Flan: FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS
- Flan-T5: Scaling Instruction-Finetuned Language Models
- Instruct-GPT: Training language models to follow instructions with human feedback
- T0: MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION
- k-INSTRUCT: SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Tasks

### Train for Dialogue
- LaMDA: Language Models for Dialog Applications
- Sparrow: Improving alignment of dialogue agents via targeted human judgements

### Chain of Thought
- Chain of Thought Prompting Elicits Reasoning in Large Language Models  :star2::star2::star2::star2::star2:
- COMPLEXITY-BASED PROMPTING FOR MULTI-STEP REASONING
- SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHT REASONING IN LANGUAGE MODELS
- Large Language Models are Zero-Shot Reasoners
- PaLM: Scaling Language Modeling with Pathways

## Resources
### paper List
- https://github.com/dongguanting/In-Context-Learning_PaperList
- https://github.com/thunlp/PromptPapers
- https://github.com/Timothyxxx/Chain-of-ThoughtsPapers

### Recommend Blog
- OpenAI ChatGPT Intro：https://openai.com/blog/chatgpt/
- OpenAI InstructGPT intro: https://openai.com/blog/instruction-following/
- AllenAI ChatGPT能力解读：[How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)  :star2::star2::star2::star2::star2:
- Huggingface ChatGPT能力解读：[The techniques behind ChatGPT: RLHF, IFT, CoT, Red teaming, and more](https://huggingface.co/blog/dialog-agents)
- Stephen Wolfram ChatGPT能力解读: [What Is ChatGPT Doing and Why Does It Work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)

### Playground
- [openAI](https://openai.com/api/): openAI API各种访问教程请自寻~
- [WriteSonic](https://app.writesonic.com/library/88c3717f-1c78-4625-bc35-055e87f05f3d/all)：AI写作，支持对话，自由创作和定向创作如广告文案，商品描述, 感觉和davinci-003效果较为相似，不过支持Web最新信息检索是亮点，支持中文
- [copy.ai](https://www.copy.ai/): WriteSonic竞品，亮点是像论文引用一样每句话都有对应网站链接，可以一键复制到右边的创作Markdown，超级好用！ :star2::star2::star2::star2::star2:
- [copy.down](https://copyai.cn/): 中文的营销文案生成，只能定向创作，支持关键词到文案的生成
- [NotionAI](https://www.notion.so/product?fredir=1)：智能Markdown，还在探索中
- [AI Topiah](https://www.ai-topia.com/): 聆心智能AI角色聊天，和路飞唠了两句，多少有点中二之魂在燃烧
- [Moss](https://moss.fastnlp.top/#/): 复旦Moss，维护中但我从来没成功打开过网页。。。

### Opensource Model
- Chatyuan: https://github.com/clue-ai/ChatYuan
- PromptCLUE: https://github.com/clue-ai/PromptCLUE
- 达摩院PLUG: https://www.alice-mind.com/portal#/
- 智源CPM2.0：https://baai.ac.cn/


