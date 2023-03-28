# DecryptPrompt

持续更新以下内容，Star to keep updated~
1. Prompt和LLM相关论文按细分方向梳理
2. AIGC相关应用
3. Prompt指南和教程
4. ChatGPT及AGI相关解读
5. 开源大模型
6. ChatGPT相关商业应用 [WIP]

## My blogs
- [解密Prompt系列1. Tunning-Free Prompt：GPT2 & GPT3 & LAMA & AutoPrompt](https://cloud.tencent.com/developer/article/2215545?areaSource=&traceId=)
- [解密Prompt系列2. 冻结Prompt微调LM： T5 & PET & LM-BFF](https://cloud.tencent.com/developer/article/2223355?areaSource=&traceId=)
- [解密Prompt系列3. 冻结LM微调Prompt: Prefix-tuning & Prompt-tuning & P-tuning](https://cloud.tencent.com/developer/article/2237259?areaSource=&traceId=)
- [解密Prompt系列4. 升级Instruction Tuning：Flan/T0/InstructGPT/TKInstruct](https://cloud.tencent.com/developer/article/2245094?areaSource=&traceId=)

## Resources 
### Tools & Tutorial
- [Langchain](https://github.com/hwchase17/langchain): 疯狂打call!!! 封装了OpenAI等多个模型的prompt链式处理能力，以及索引构建，prompt生成等集成能力。以下DocsGPT等多个应用都是基于langchain开发，虽然框架写的略显复杂，不过能力很完善！Demo很丰富！如果你想基于ChatGPT快速实现一些类似文档QA，Bing搜索一类的集成方案，它是你的不二选择 :star::star:
- [openAI](https://openai.com/api/): ChatGPT出API啦, 价格下降10倍！
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook): 提供OpenAI模型使用示例  :star:
- [OpenAI 接口被墙解决办法](https://github.com/riba2534/openai-scf-goproxy): 使用腾讯云搭建代理，亲测非常好用且手残党也可以轻松上手
- [PromptPerfect](https://promptperfect.jinaai.cn/):用魔法打败魔法，输入原始提示词，模型进行定向优化，试用后我有点沉默了，可以定向支持不同使用prompt的模型如Difussion，ChatGPT， Dalle等
- [ClickPrompt](https://www.clickprompt.org/zh-CN/): 为各种prompt加持的工具生成指令包括Difussion，chatgptdeng, 需要OpenAI Key 
- [ChatGPT ShortCut](https://newzone.top/chatgpt/)：提供各式场景下的Prompt范例，范例很全，使用后可以点赞！  :star:
- [Full ChatGPT Prompts + Resources](https://enchanting-trader-463.notion.site/Full-ChatGPT-Prompts-Resources-8aa78bb226b7467ab59b70d2b27042e9): 各种尝尽的prompt范例，和以上场景有所不同
- [learning Prompt](https://learnprompting.org/):  prompt engineering超全教程，和落地应用收藏，几乎包括以上内容，不过因为太全了，想要找到想要的内容有些难度。
- [Prompt-Engineer-Guide]( https://github.com/dair-ai/Prompt-Engineering-Guide): 同learnig prompt类的集成教程，互相引用可还行？！分类索引做的更好些 :star:
- [OpenAI 应用汇总指南](https://www.mojidoc.com/05z7y-dd5pa7hu3zfmhnbngoeztyqcnq-00b): 纯应用类的汇总指南

### AIGC playground
- [New Bing](https://www.bing.com/)：需要连外网否则会重定向到bing中国，需要申请waitlist ![](https://img.shields.io/badge/AIGC-Search-yellow) :star:
- [DocsGPT](https://github.com/arc53/DocsGPT): 把ChatGPT开放域问答转化成封闭域问答的通用方案，试用垂类领域问答场景,可以试用定制的ChatBot ![](https://img.shields.io/badge/AIGC-Chatbot-blue) :star:
- [ChatPDF/Chat2Doc](https://chat2doc.cn/): 国内的ChatPDF, 上传pdf后，会给出文章的Top5可能问题，然后对话式从文档中进行问答和检索，10s读3万字 ![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [ChatPaper](https://github.com/kaixindelele/ChatPaper): 根据输入关键词，自动在arxiv上下载最新的论文，并对论文进行摘要总结，可以在huggingface上试用！![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [researchgpt](https://github.com/mukulpatnaik/researchgpt): 和ChatPDF类似，支持arivx论文下载，加载后对话式获取论文重点 ![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [AI Topiah](https://www.ai-topia.com/): 聆心智能AI角色聊天，和路飞唠了两句，多少有点中二之魂在燃烧 ![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [chatbase](https://www.chatbase.co/): 情感角色聊天，还没尝试 ![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [Vana](https://gptme.vana.com/login): virtual DNA, 通过聊天创建虚拟自己！概念很炫  ![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [WriteSonic](https://app.writesonic.com/)：AI写作，支持对话和定向创作如广告文案，商品描述, 支持Web检索是亮点，支持中文  ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [copy.ai](https://www.copy.ai/): WriteSonic竞品，亮点是像论文引用一样每句话都有对应网站链接，可以一键复制到右边的创作Markdown，超级好用！ ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen) :star:
- [NotionAI](https://www.notion.so/product?fredir=1)：智能Markdown，适用真相！在创作中用command调用AI辅助润色，扩写，检索内容，给创意idea ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [Jasper](https://www.jasper.ai/): 同上，全是竞品哈哈  ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [copy.down](https://copyai.cn/): 中文的营销文案生成，只能定向创作，支持关键词到文案的生成  ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [ChatExcel](https://chatexcel.com/convert): 指令控制excel计算，对熟悉excel的有些鸡肋，对不熟悉的有点用  ![](https://img.shields.io/badge/Tool-Business-red)
- [ChatPPT](https://github.com/williamfzc/chat-gpt-ppt): 使用ChatGPT进行PPT制作 ![](https://img.shields.io/badge/Tool-Business-red)
- [BibiGPT](https://github.com/JimmyLv/BibiGPT): Bilibli视频内容一键总结，多模态文档  ![](https://img.shields.io/badge/Tool-Business-red)
- Microsoft 365 Copilot：微软Office全面接入GPT4，智能PPT，Excel，Word，暂无链接。其实就是上面开源创意的全家桶套餐 ![](https://img.shields.io/badge/Tool-Business-red)
- Google Workspace: 谷歌推出的搭载各种AI服务的办公场景全覆盖，暂无使用方案。![](https://img.shields.io/badge/Tool-Business-red)
- [Copilot](https://github.com/features/copilot): 要付费哟 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [Fauxpilot](https://github.com/fauxpilot/fauxpilot): copilot本地开源替代 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [CodeGex](http://codegeex.cn/zh-CN): 国内替代品，还没试过 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [dreamstudio.ai](https://beta.dreamstudio.ai/dream): 开创者，Stable Difussion， 有试用quota ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F): 开创者，艺术风格为主 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [Dall.E](https://openai.com/product/dall-e-2): 三巨头这就凑齐了 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [ControlNet](https://huggingface.co/spaces/hysts/ControlNet): 为绘画创作加持可控性 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [GFPGAN](https://github.com/Nutlope/restorePhotos): 照片修复  ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [Visual ChatGPT](https://huggingface.co/spaces/microsoft/visual_chatgpt): 微软发布图像ChatGPT，对话方式进行图像生成编辑，问答 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange) :star:

### 相关模型
#### 国外
- [Google Bard](https://bard.google.com): 谷歌bard虽迟但到，可以申请waitlist了
- [LLaMA](https://github.com/facebookresearch/llama):Meta开源指令微调LLM，规模70 亿到 650 亿不等
- [ChatLLaMA](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama): 基于RLHF微调了LLaMA 
- [PaLM-E](https://palm-e.github.io): 谷歌多模态大模型，540B的PaLM语言模型和22B的ViT视觉模型相结合，得到562B的PaLM-E模型，在机器人应用场景有了新的突破
- [MetaLM](https://github.com/microsoft/unilm): 微软开源的大规模自监督预训练模型
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca): 斯坦福开源的使用52k数据在7B的LLaMA上微调得到，据说效果类似text-davinci-003, 模型不久后会发布
- [OPT-IML](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/metaseq/tree/main/projects/OPT): Meta复刻GPT3，up to 175B, 不过效果并不及GPT3
- [Bloom](https://huggingface.co/bigscience/bloom)：BigScience出品，规模最大176B, 感觉应该对标text-davinci-002
- [T0](https://github.com/bigscience-workshop/t-zero): BigScience出品，3B~11B的在T5进行指令微调的模型


#### 国内
- [文心一言](https://yiyan.baidu.com/welcome):已经拿到邀请码并试用，虽然人格化程度显著低，但效果上并没有很拉胯，国产YYDS！3.31号API就开放使用了，期待ing
- [ChatGLM](https://github.com/THUDM/ChatGLM-6B): 清华开源的、支持中英双语的对话语言模型，使用了代码训练，指令微调和RLHF。和以下GLM相同大小的130B的模型还在开发中。局限性是有的，不过这个模型大小很合适，准备改天试一下看看
- [Moss](https://moss.fastnlp.top/#/): 复旦发布的大模型
- https://www.modelscope.cn/home：国内开源模型魔塔社区
- [PromptCLUE](https://github.com/clue-ai/PromptCLUE): 多任务Prompt语言模型
- [Chatyuan](https://github.com/clue-ai/ChatYuan)：基于PromptCLUE训练的对话模型
- [PLUG](https://www.alice-mind.com/portal#/): 阿里达摩院发布的大模型，提交申请会给下载链接
- [CPM2.0](https://baai.ac.cn/): 智源发布CPM2.0
- [GLM](https://github.com/THUDM/GLM-130B): 清华发布的中英双语130B大模型

### Recommend Blog
- [OpenAI ChatGPT Intro](https://openai.com/blog/chatgpt/)
- [OpenAI InstructGPT intro](https://openai.com/blog/instruction-following/)
- AllenAI ChatGPT能力解读：[How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)  :star:
- Huggingface ChatGPT能力解读：[The techniques behind ChatGPT: RLHF, IFT, CoT, Red teaming, and more](https://huggingface.co/blog/dialog-agents)
- Stephen Wolfram ChatGPT能力解读: [What Is ChatGPT Doing and Why Does It Work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)
- [Chatgpt相关解读汇总](https://github.com/chenweiphd/ChatGPT-Hub)
- [麻省理工科技采访OpenAI工程师](https://www.technologyreview.com/2023/03/03/1069311/inside-story-oral-history-how-chatgpt-built-openai/)
- [AGI历史与现状](https://www.jiqizhixin.com/articles/2018-11-15-6?from=timeline)
- [张俊林 通向AGI之路：大型语言模型（LLM）技术精要](https://zhuanlan.zhihu.com/p/597586623)
- [知乎回答 OpenAI 发布 GPT-4，有哪些技术上的优化或突破?](https://www.zhihu.com/question/589639535/answer/2936696161)
- [追赶ChatGPT的难点与平替](https://zhuanlan.zhihu.com/p/609877277)

### ChatGPT 其他商用场景
1. shopify：私人导购，情人节给女友买点啥？问它
3. Instcart：私人营养师，搭配食谱，相关商品直接加购物车
4. Quizlet：私人教辅，你来学习，它来出题，帮你答疑
5. HSBC：使用大模型在财务信息汇总和分类
6. Soul：塑造数字人，并为媒体和娱乐产品自动生成内容
7. Salesforce：旗下的 Slack宣布了一款新的人工智能应用程序，它将在几秒钟内回复同事消息，并进行会议总结


### paper List
- https://github.com/dongguanting/In-Context-Learning_PaperList
- https://github.com/thunlp/PromptPapers
- https://github.com/Timothyxxx/Chain-of-ThoughtsPapers

## Papers
### Survey
- Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing :star:
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
- GPT3: Language Models are Few-Shot Learners   :star:
- LAMA: Language Models as Knowledge Bases?
- AutoPrompt: Eliciting Knowledge from Language Models

### Fix-Prompt LM Tunning
- T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- PET-TC(a): Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference  :star:
- PET-TC(b): PETSGLUE It’s Not Just Size That Matters Small Language Models are also few-shot learners
- GenPET: Few-Shot Text Generation with Natural Language Instructions
- LM-BFF: Making Pre-trained Language Models Better Few-shot Learners  :star:
- ADEPT: Improving and Simplifying Pattern Exploiting Training

### Fix-LM Prompt Tunning 
- Prefix-tuning: Optimizing continuous prompts for generation  
- Prompt-tunning: The power of scale for parameter-efficient prompt tuning :star:
- P-tunning: GPT Understands Too :star:
- WARP: Word-level Adversarial ReProgramming

### LM + Prompt Tunning 
- P-tunning v2: Prompt Tuning Can Be Comparable to Fine-tunning Universally Across Scales and Tasks
- PTR: Prompt Tuning with Rules for Text Classification
- PADA: Example-based Prompt Learning for on-the-fly Adaptation to Unseen Domains

### Instruction Tunning LLMs 
- Flan: FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS :star:
- Flan-T5: Scaling Instruction-Finetuned Language Models
- Instruct-GPT: Training language models to follow instructions with human feedback star:
- T0: MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION
- Tk-INSTRUCT: SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Tasks

### Train for Dialogue
- LaMDA: Language Models for Dialog Applications
- Sparrow: Improving alignment of dialogue agents via targeted human judgements star:
- BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage
- How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation

### Chain of Thought
- Chain of Thought Prompting Elicits Reasoning in Large Language Models  :star:
- COMPLEXITY-BASED PROMPTING FOR MULTI-STEP REASONING
- SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHT REASONING IN LANGUAGE MODELS
- Large Language Models are Zero-Shot Reasoners
- PaLM: Scaling Language Modeling with Pathways

### RLHF
- Deepmind
  - Teaching language models to support answers with verified quotes
  - sparrow, Improving alignment of dialogue agents via targetd human judgements
- openai
  - PPO: Proximal Policy Optimization Algorithms :star:
  - Deep Reinforcement Learning for Human Preference
  - Fine-Tuning Language Models from Human Preferences
  - learning to summarize from human feedback
  - InstructGPT: Training language models to follow instructions with human feedback :star:
- Anthropic
  - Red Teaming Language Models to Reduce Harms Methods,Scaling Behaviors and Lessons Learned
  - Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
  - Constitutional AI Harmlessness from AI Feedback :star:
- AllenAI, RL4LM：IS REINFORCEMENT LEARNING (NOT) FOR NATURAL LANGUAGE PROCESSING BENCHMARKS


### Agent: 让模型使用工具
- Tool Former: Toolformer: Language Models Can Teach Themselves to Use Tools
- MRKL SystemsA modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning
- ReAct: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS
- Self: MEASURING AND NARROWING THE COMPOSITIONALITY GAP IN LANGUAGE MODELS
- PAL: Program-aided Language Models
