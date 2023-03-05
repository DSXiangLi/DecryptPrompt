# DecryptPrompt

试(努)图(力)理解ChatGPT的超能力来源，顺手梳理下Prompt范式相关模型，持续更新中，有推荐paper和resources欢迎PR哟~~

每个方向推荐1-2篇五星好评的论文，没有推荐的方向就是我也还没看完哈哈哈~

补充AIGC相关的应用

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
- Flan: FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS :star2::star2::star2::star2::star2:
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

### RLHF
- Deep reinforcement learning from human preferences
- PPO: Proximal Policy Optimization Algorithms :star2::star2::star2::star2::star2:
- InstrutGPT序作：learning to summarize from human feedback
- InstructGPT: Training language models to follow instructions with human feedback
- RL4LM：IS REINFORCEMENT LEARNING (NOT) FOR NATURAL LANGUAGE PROCESSING BENCHMARKS

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
- https://github.com/chenweiphd/ChatGPT-Hub: Chatgpt相关解读汇总

### AIGC Playground
- [openAI](https://openai.com/api/): ChatGPT出API啦, 价格下降10倍！ ![](https://img.shields.io/badge/AIGC-Chatbot-blue) 
- [AI Topiah](https://www.ai-topia.com/): 聆心智能AI角色聊天，和路飞唠了两句，多少有点中二之魂在燃烧 ![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [chatbase](https://www.chatbase.co/): 情感角色聊天，还没尝试 ![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [New Bing](https://www.bing.com/)：需要连外网否则会重定向到bing中国，需要申请waitlist ![](https://img.shields.io/badge/AIGC-Search-yellow)
- [WriteSonic](https://app.writesonic.com/)：AI写作，支持对话和定向创作如广告文案，商品描述, 支持Web检索是亮点，支持中文  ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [copy.ai](https://www.copy.ai/): WriteSonic竞品，亮点是像论文引用一样每句话都有对应网站链接，可以一键复制到右边的创作Markdown，超级好用！ ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [NotionAI](https://www.notion.so/product?fredir=1)：智能Markdown，适用真相！在创作中用command调用AI辅助润色，扩写，检索内容，给创意idea ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [Jasper](https://www.jasper.ai/): 同上，全是竞品哈哈  ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [copy.down](https://copyai.cn/): 中文的营销文案生成，只能定向创作，支持关键词到文案的生成  ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [Copilot](https://github.com/features/copilot): 要付费哟 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [Fauxpilot](https://github.com/fauxpilot/fauxpilot): copilot本地开源替代 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [CodeGex](http://codegeex.cn/zh-CN): 国内替代品，还没试过 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [dreamstudio.ai](https://beta.dreamstudio.ai/dream): 开创者，Stable Difussion， 有试用quota ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F): 开创者，艺术风格为主 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [Dall.E](https://openai.com/product/dall-e-2): 三巨头这就凑齐了 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)

### Opensource Model
#### 国外
- [OPT-IML](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/metaseq/tree/main/projects/OPT): Meta复刻GPT3，up to 175B,不过效果并不及GPT3
- [Bloom](https://huggingface.co/bigscience/bloom)：BigScience复刻,up to 176B, 搂了一眼感觉应该对标text-davinci-002

#### 国内
- 国内开源模型魔塔社区：https://www.modelscope.cn/home
- Chatyuan: https://github.com/clue-ai/ChatYuan
- PromptCLUE: https://github.com/clue-ai/PromptCLUE
- 达摩院PLUG: https://www.alice-mind.com/portal#/
- 智源CPM2.0：https://baai.ac.cn/
- Moss：https://moss.fastnlp.top/#/，我还没成功打开过网页。。



