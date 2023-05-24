# DecryptPrompt
> 如果LLM的突然到来让你感到沮丧，不妨读下主目录的Choose Your Weapon Survival Strategies for Depressed AI Academics
持续更新以下内容，Star to keep updated~
1. 开源LLM
2. 指令微调和RLHF数据以及训练框架
3. Prompt和LLM相关论文按细分方向梳理
4. AIGC相关应用
5. Prompt指南和教程
6. ChatGPT及AGI相关解读

## My blogs & ChatGPT应用
- [解密Prompt系列1. Tunning-Free Prompt：GPT2 & GPT3 & LAMA & AutoPrompt](https://cloud.tencent.com/developer/article/2215545?areaSource=&traceId=)
- [解密Prompt系列2. 冻结Prompt微调LM： T5 & PET & LM-BFF](https://cloud.tencent.com/developer/article/2223355?areaSource=&traceId=)
- [解密Prompt系列3. 冻结LM微调Prompt: Prefix-tuning & Prompt-tuning & P-tuning](https://cloud.tencent.com/developer/article/2237259?areaSource=&traceId=)
- [解密Prompt系列4. 升级Instruction Tuning：Flan/T0/InstructGPT/TKInstruct](https://cloud.tencent.com/developer/article/2245094?areaSource=&traceId=)
- [解密prompt系列5. APE+SELF=自动化指令集构建代码实现](https://cloud.tencent.com/developer/article/2260697?areaSource=&traceId=)
- [解密Prompt系列6. lora指令微调扣细节-请冷静,1个小时真不够~](https://cloud.tencent.com/developer/article/2276508)
- [解密Prompt系列7. 偏好对齐RLHF-OpenAI·DeepMind·Anthropic对比分析](https://cloud.tencent.com/developer/article/old/2289566?areaSource=&traceId=)
- [ChatGPT应用1. MakeInstruction零人工指令样本构建](https://huggingface.co/spaces/xl2533/MakeInstruction)
- [ChatGPT应用2. ChatPDF简单复现](https://huggingface.co/spaces/xl2533/FinDoc)


## 模型和数据
- [高星中文「类Alpaca」项目对比测评](https://zhuanlan.zhihu.com/p/625195238)
- [可商用LLM列表](https://github.com/eugeneyan/open-llms)
- [CMU开源聊天机器人评测应用](https://github.com/zeno-ml/zeno-build): ChatGPT>Vicuna>others；在对话场景中训练可能很重要
- [Berkley出品大模型排位赛榜有准中文榜单](https://lmsys.org/blog/2023-05-03-arena/): GPT4自然是稳居第一，GPT4>Claude>GPT3.5>Vicuna>others

### 国外模型
|模型链接     | 模型描述    |
| --- | --- |
| [Google Bard](https://bard.google.com)   |    谷歌bard虽迟但到，可以申请waitlist了 |
|  [Claude](https://www.anthropic.com/product)   |  ChatGPT最大竞争对手Claude也开放申请了，slack中无限试用   |
| [LLaMA](https://github.com/facebookresearch/llama)    |  Meta开源指令微调LLM，规模70 亿到 650 亿不等  |
|[MPT](https://huggingface.co/mosaicml/mpt-7b-chat)|MosaicML开源的预训练+指令微调的新模型，可商用，支持84k tokens超长输入|
|[RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1)|RedPajama项目既开源预训练数据后开源3B，7B的预训练+指令微调模型|
|[ChatLLaMA](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama)     | 基于RLHF微调了LLaMA     |
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)    |  斯坦福开源的使用52k数据在7B的LLaMA上微调得到，   |
|[Alpaca-lora](https://github.com/tloen/alpaca-lora)     |   LORA微调的LLaMA  |
|[Dromedary](https://github.com/IBM/Dromedary)|IBM self-aligned model with the LLaMA base|
|[Vicuna](https://github.com/lm-sys/FastChat)|Alpaca前成员等开源以LLama13B为基础使用ShareGPT指令微调的模型，提出了用GPT4来评测模型效果|
|[ColossalChat](https://github.com/hpcaitech/ColossalAI)|HPC-AI Tech开源的Llama+RLHF微调|
|[MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)|Vicuna+BLIP2 文本视觉融合|
|[StackLLama](https://huggingface.co/trl-lib/llama-7b-se-rl-peft)|LLama使用Stackexchange数据+SFT+RL|
|[Cerebras](https://huggingface.co/cerebras/Cerebras-GPT-13B)|Cerebras开源了1亿到130亿的7个模型，从预训练数据到参数全开源|
|[PaLM-E](https://palm-e.github.io)     |  谷歌多模态大模型，540B的PaLM语言模型和22B的ViT视觉模型相结合，得到562B的PaLM-E模型，在机器人应用场景有了新的突破   |
|[Dolly-v2](https://huggingface.co/databricks/dolly-v2-7b)|可商用 7b指令微调开源模型在GPT-J-6B上微调|
|[OpenChatKit](https://github.com/togethercomputer/OpenChatKit)|openai研究员打造GPT-NoX-20B微调+6B审核模型过滤|
|  [MetaLM](https://github.com/microsoft/unilm)    | 微软开源的大规模自监督预训练模型    |
|[Amazon Titan](https://aws.amazon.com/cn/bedrock/titan/)|亚马逊在aws上增加自家大模型|
| [OPT-IML](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/metaseq/tree/main/projects/OPT)     |  Meta复刻GPT3，up to 175B, 不过效果并不及GPT3   |
|[Bloom](https://huggingface.co/bigscience/bloom)|BigScience出品，规模最大176B|
|[BloomZ](https://huggingface.co/bigscience/bloomz)|BigScience出品, 基于Bloom微调|
|[Galacia](https://github.com/paperswithcode/galai)|和Bloom相似，更针对科研领域训练的模型|
| [T0](https://github.com/bigscience-workshop/t-zero)|BigScience出品，3B~11B的在T5进行指令微调的模型|

### 国内模型
|模型链接     | 模型描述    |
| --- | --- |
|  [ChatGLM](https://github.com/THUDM/ChatGLM-6B)   |     清华开源的、支持中英双语的对话语言模型，使用了代码训练，指令微调和RLHF。和以下GLM相同大小的130B的模型还在开发中。试用了下超出预期！|
|  [Moss](https://github.com/OpenLMLab/MOSS)   |  为复旦正名！开源了预训练，指令微调的全部数据和模型。可商用 |
|[Wombat-7B](https://huggingface.co/GanjinZero/wombat-7b-delta)|达摩院开源无需强化学习使用RRHF对齐的语言模型, alpaca基座|
|[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)     |   哈工大中文指令微调的LLaMA  |
|  [Luotuo](https://github.com/LC1332/Luotuo-Chinese-LLM)   |  中文指令微调的LLaMA，和ChatGLM   |
| [文心一言](https://yiyan.baidu.com/welcome)    |  已经拿到邀请码并试用，虽然人格化程度显著低，但效果上并没有很拉胯，国产YYDS！不过商业化霸王条款确实不少   |
|[通义千问](https://tongyi.aliyun.com/)|阿里系LLM开放申请|
|[星火](https://passport.xfyun.cn/login)|科大讯飞星火，数学是真的厉害|
|[BiLLa](https://github.com/Neutralzz/BiLLa)|LLama词表扩充预训练+预训练和任务1比1混合SFT+指令样本SFT三阶段训练|
|[Phoenix](https://github.com/FreedomIntelligence/LLMZoo)|港中文开源凤凰和奇美拉LLM，Bloom基座，40+语言支持|
|[Guanaco](https://huggingface.co/KBlueLeaf/guanaco-7B-leh)|LLama 7B基座，在alpaca52K数据上加入534K多语言指令数据微调|
|[ziya](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-7B-Reward)|IDEA研究院在7B/13B llama上继续预训练+SFT+RM+PPO+HFTT+COHFT+RBRS|
|[Chinese Vincuna](https://github.com/Facico/Chinese-Vicuna)|LLama 7B基座，使用Belle+Guanaco数据训练|
|[Linly](https://github.com/CVI-SZU/Linly)|Llama 7B基座，使用belle+guanaco+pclue+firefly+CSL+newscommentary等7个指令微调数据集训练|
|[Firefly](https://github.com/yangjianxin1/Firefly)| 中文2.6B模型，提升模型中文写作，古文能力，待开源全部训练代码，当前只有模型|
| [Baize](https://github.com/project-baize/baize-chatbot)    | 使用100k self-chat对话数据微调的LLama    |
| [BELLE](https://github.com/LianjiaTech/BELLE)    |使用ChatGPT生成数据对开源模型进行中文优化  |
|[Chatyuan](https://github.com/search?q=chatyuan&type=repositories)|chatgpt出来后最早的国内开源对话模型，T5架构是下面PromptCLUE的衍生模型|
| [PromptCLUE](https://github.com/clue-ai/PromptCLUE)    | 多任务Prompt语言模型    |
| [PLUG](https://www.alice-mind.com/portal#/)    |   阿里达摩院发布的大模型，提交申请会给下载链接  |
|[CPM2.0](https://baai.ac.cn/)     |  智源发布CPM2.0    |
|[GLM](https://github.com/THUDM/GLM-130B) |   清华发布的中英双语130B预训练模型 |

### 垂直领域模型
|模型链接     | 模型描述  
| --- | --- | 
|MedPalm|Google在Faln-PaLM的基础上通过多种类型的医疗QA数据进行prompt-tuning指令微调得到，同时构建了MultiMedQA|
|[ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)|110K真实医患对话样本+5KChatGPT生成数据进行指令微调|
|[Huatuo](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) [Med-ChatGLM](https://github.com/SCIR-HI/Med-ChatGLM)|医学知识图谱和chatgpt构建中文医学指令数据集+医学文献和chatgpt构建多轮问答数据|
|[Chinese-vicuna-med](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-medical.md)|Chinese-vicuna在cMedQA2数据上微调|
|[OpenBioMed](https://github.com/BioFM/OpenBioMed)|清华AIR开源轻量版BioMedGPT, 知识图谱&20+生物研究领域多模态预训练模型|
|[DoctorGLM](https://github.com/xionghonglin/DoctorGLM)|ChatDoctor+MedDialog+CMD 多轮对话+单轮指令样本微调GLM|
|[MedicalGPT-zh](https://github.com/MediaBrain-SJTU/MedicalGPT-zh)|自建的医学数据库ChatGPT生成QA+16个情境下SELF构建情景对话|
|[PMC-LLaMA](https://github.com/chaoyi-wu/PMC-LLaMA)|医疗论文微调Llama|
|[ NHS-LLM](https://github.com/CogStack/OpenGPT/tree/main)|Chatgpt生成的医疗问答，对话，微调模型|
|[LawGPT-zh](https://github.com/LiuHC0428/LAW-GPT)|利用ChatGPT清洗CrimeKgAssitant数据集得到52k单轮问答+我们根据中华人民共和国法律手册上最核心的9k法律条文，利用ChatGPT联想生成具体的情景问答+知识问答使用ChatGPT基于文本构建QA对|
|[FinChat.io](https://finchat.io/)|使用最新的财务数据，电话会议记录，季度和年度报告，投资书籍等进行训练|
|[OpenGPT](https://github.com/CogStack/OpenGPT)|领域LLM指令样本生成+微调框架|

### 指令微调&RL工具
| 工具描述   | 链接   | 
| --- | --- | 
|LoRA：Low-Rank指令微调方案|https://github.com/tloen/alpaca-lora|
|peft：parameter-efficient prompt tunnging工具集|https://github.com/huggingface/peft|
|RL4LMs：AllenAI的RL工具|https://github.com/allenai/RL4LMs|
|trl：基于Transformer的强化训练框架|https://github.com/lvwerra/trl|
|trlx：分布式训练trl | https://github.com/CarperAI/trlx|
|北大开源河狸项目可复现RLHF，支持多数LLM，提供RLHF数据|https://github.com/PKU-Alignment/safe-rlhf|
|RL4LMs：AllenAI的RL工具|https://github.com/allenai/RL4LMs|
|LMFlow：港科大实验室开源的大模型微调框架，支持以上多数开源模型的指令微调和RLHF|https://github.com/OptimalScale/LMFlow|
|hugNLP:基于Huggingface开发继承Prompt技术，预训练和是指输入等多种方案|https://github.com/wjn1996/HugNLP|
|Deepspeed：针对RL训练和推理的整合优化|https://github.com/microsoft/DeepSpeed|
|Uerpy:预训练框架支持lm,mlm,unilm等|https://github.com/dbiir/UER-py|
|TecentPretrain: Uerpy的重构版本支持llama预训练|https://github.com/Tencent/TencentPretrain/tree/main|
|langchain：LLM工具集|https://github.com/hwchase17/langchain|
|BMTTools: 清华出品类似langchain|https://github.com/OpenBMB/BMTools|
|BabyAGI：自执行LLM Agent|https://github.com/yoheinakajima/babyagi|
|AutoGPT：自执行LLM Agent|https://github.com/Torantulino/Auto-GPT|
|Jarvis: 大模型调用小模型框架，给小模型一个未来！|https://github.com/search?q=jarvis|
|lamini: 整合指令数据生成，SFT，RLHF的工具库|https://github.com/lamini-ai/lamini/|
|wenda:闻达小模型整合搜索用于知识融入|https://github.com/l15y/wenda|
|Chain-of-thought-hub：模型推理能力评估平台|https://github.com/FranxYao/chain-of-thought-hub|
|FlexGen:LLM推理 CPU Offload计算架构|https://github.com/FMInference/FlexGen|
### 开源数据
| 数据类型    | 数据描述    | 数据链接    |
| --- | --- | --- |
|  指令微调   |  self-instruct，GPT3自动生成&过滤得到指令集   |   https://github.com/yizhongw/self-instruct   |
|  指令微调      |    Standford Alpaca：52K text-davinci-003生成的self-instruct指令数据集 |  https://github.com/tatsu-lab/stanford_alpaca   |
|  指令微调      | 中文翻译Alpaca还有一些其他指令数据集    |    https://github.com/hikariming/alpaca_chinese_dataset https://github.com/carbonz0/alpaca-chinese-dataset|
|指令微调|alpaca指令GPT4生成，和以上几版对比显著质量更高，回复更长|https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/tree/main|
|  指令微调      |   Guanaco数据：对Alphca指令重写后以不同语言生成总共534K，有对话和非对话类型，还有补充的QA生成样本  | https://huggingface.co/datasets/JosephusCheung/GuanacoDataset    |
|指令微调|OIG中文指令包括翻译alpaca+natural+unnatural，多轮对话，考试，leetcode指令|https://github.com/BAAI-Zlab/COIG|
|指令微调|Vicuna训练使用的样本，用API获取了sharegpt上用户和chatgpt对话历史，部分网友整理到了HF|https://github.com/domeccleston/sharegpt https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main|
|指令微调|HC3指令数据中英文，包括金融，开放QA，百科，DBQA，医学等包含人工回复|https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/tree/main|
|指令微调|MOSS开源的SFT数据包含使用plugin的对话数据|https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/tree/main|
|指令微调|InstructWild数据：用四处爬取的chatgpt指令作为种子self-instruct扩充生成，中英双语|https://github.com/XueFuzhao/InstructionWild/tree/main/data|
|  指令微调      |  BELLE100万指令数据，参考Alpaca用ChatGPT生成，有数学，多轮对话，校色对话等等   |   https://github.com/LianjiaTech/BELLE  |
|  指令微调      | PromptCLUE多任务提示数据集：模板构建，只包含标准NLP任务     |  https://github.com/CLUEbenchmark/pCLUE   |
|  指令微调      |  TK-Instruct微调用的指令数据集, 全人工标注1600+NLP任务   |   https://instructions.apps.allenai.org/  |
|   指令微调     | T0微调用的指令数据集（P3）    |  https://huggingface.co/datasets/bigscience/P3   |
|  指令微调   | p3衍生的46种多语言数据集（xmtf）    | https://github.com/bigscience-workshop/xmtf    |
|  指令微调      |Unnatural Instruction使用GPT3生成后改写得到240k     |   https://github.com/orhonovich/unnatural-instructions  |
|指令微调|alpaca COT对多个数据源进行了清理并统一格式放到的了HF, 重点是人工整理的COT数据|https://github.com/PhoebusSi/Alpaca-CoT|
|指令微调|人工编写包含23种常见的中文NLP任务的指令数据，中文写作方向|https://github.com/yangjianxin1/Firefly|
|指令微调|Amazon COT指令样本包括各类QA，bigbench，math等|https://github.com/amazon-science/auto-cot|
|指令微调|CSL包含 396,209 篇中文核心期刊论文元信息 （标题、摘要、关键词、学科、门类）可做预训练可构建NLP指令任务|https://github.com/ydli-ai/CSL|
|对话指令|LAION 策划的开放指令通用数据集中手动选择的组件子集 已开源40M 3万个,100M在路上 |https://github.com/LAION-AI/Open-Instruction-Generalist|
|  对话指令      |Baize基于Chat GPT构建的self-chat数据    |   https://github.com/project-baize/baize-chatbot/tree/main/data   |
|  对话指令      |FaceBook开源BlenderBot训练对话数据~6K     |   https://huggingface.co/datasets/blended_skill_talk   |
|  对话指令      |AllenAI开源38.5万个对话高质量数据集SODA   |   https://realtoxicityprompts.apps.allenai.org/ |
|  对话指令      |InstructDial在单一对话任务类型上进行指令微调   |  https://github.com/prakharguptaz/Instructdial   |
| 对话指令 |Ultra Chat 两个独立的 ChatGPT Turbo API 进行对话，从而生成多轮对话数据|https://github.com/thunlp/UltraChat|
|对话指令|Awesome Open-domain Dialogue Models提供多个开放域对话数据|https://github.com/cingtiye/Awesome-Open-domain-Dialogue-Models#%E4%B8%AD%E6%96%87%E5%BC%80%E6%94%BE%E5%9F%9F%E5%AF%B9%E8%AF%9D%E6%95%B0%E6%8D%AE%E9%9B%86|
|RLFH|北大河狸开源RLHF数据集10K，1M需要申请|https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K|
|  RLHF     |  Anthropic hh-rlhf数据集   |   https://huggingface.co/datasets/Anthropic/hh-rlhf  |
|RLHF|Stack-exchange上问题对应多个答案，每个答案有打分|https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences/tree/main|
|  RLHF     | Facebook Bot Adversarial Dialogues数据集5K   |  https://github.com/facebookresearch/ParlAI  |
|  RLHF     | AllenAI Real Toxicity prompts   |  https://github.com/facebookresearch/ParlAI  |
|RLHF|OpenAssistant Conversations 160K消息，13500人工生成, 英文为主|https://huggingface.co/datasets/OpenAssistant/oasst1|
| 评估集     | BigBench(Beyond the Imitation Game Benchmark)    |  https://github.com/google/BIG-bench   |
|   评估集       |   Complex QA：用于ChatGPT的评测指令集  |    https://github.com/tan92hl/Complex-Question-Answering-Evaluation-of-ChatGPT |
|   评估集      |   Langchain开源评估数据集  |   https://huggingface.co/LangChainDatasets  |
|评估集|2010-2022年全国高考卷的题目|https://github.com/OpenLMLab/GAOKAO-Bench|
|评估集|中文通用大模型综合性评测基准SuperCLUE|https://github.com/CLUEbenchmark/SuperCLUE|
|预训练|RedPajama开源的复刻llama的预训练数据集|https://github.com/togethercomputer/RedPajama-Data|
|预训练|UER整理CLUECorpusSmall+News Commentary中英|https://github.com/dbiir/UER-py/wiki/%E9%A2%84%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE|
|预训练|智源人工智能开源的wudao 200G预训练数据|[https://github.com/BAAI-WuDao/WuDaoMM](https://openi.pcl.ac.cn/BAAI/WuDao-Data)|
|多源数据集整合|opendatalab整合了预训练阶段的多个数据源|https://opendatalab.org.cn/?industry=9821&source=JUU3JTlGJUE1JUU0JUI5JThF|

## Resources 
### Tools & Tutorial
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook): 提供OpenAI模型使用示例  :star:
- [OpenAI 接口被墙解决办法](https://github.com/riba2534/openai-scf-goproxy): 使用腾讯云搭建代理，亲测非常好用且手残党也可以轻松上手
- [PromptPerfect](https://promptperfect.jinaai.cn/):用魔法打败魔法，输入原始提示词，模型进行定向优化，试用后我有点沉默了，可以定向支持不同使用prompt的模型如Difussion，ChatGPT， Dalle等
- [ClickPrompt](https://www.clickprompt.org/zh-CN/): 为各种prompt加持的工具生成指令包括Difussion，chatgptdeng, 需要OpenAI Key 
- [ChatGPT ShortCut](https://newzone.top/chatgpt/)：提供各式场景下的Prompt范例，范例很全，使用后可以点赞！  :star:
- [Full ChatGPT Prompts + Resources](https://enchanting-trader-463.notion.site/Full-ChatGPT-Prompts-Resources-8aa78bb226b7467ab59b70d2b27042e9): 各种尝尽的prompt范例，和以上场景有所不同
- [learning Prompt](https://learnprompting.org/):  prompt engineering超全教程，和落地应用收藏，包括很多LLM调用Agent的高级场景 :star:
- [The art of asking chatgpt for high quality answers](https://github.com/ORDINAND/The-Art-of-Asking-ChatGPT-for-High-Quality-Answers-A-complete-Guide-to-Prompt-Engineering-Technique): 如何写Prompt指令出书了，链接是中文翻译的版本，比较偏基础使用
- [Prompt-Engineer-Guide]( https://github.com/dair-ai/Prompt-Engineering-Guide): 同learnig prompt类的集成教程，互相引用可还行？！分类索引做的更好些 :star:
- [OpenAI 应用汇总指南](https://www.mojidoc.com/05z7y-dd5pa7hu3zfmhnbngoeztyqcnq-00b): 纯应用类的汇总指南
- [AI 导航](https://www.ainavpro.com/#term-209): 包括但不限于ChatGPT的应用汇总网站，更新很快，发现了一些新大陆
- [AI Alignment Forum](https://www.alignmentforum.org/): RLHF等对齐相关最新论文和观点的讨论论坛

### AIGC playground
- [cognosys](https://www.cognosys.ai/create): 全网最火的web端AutoGPT，不过咋说呢试用了下感觉下巴要笑掉了，不剧透去试试你就知道 ![](https://img.shields.io/badge/Auto-Agent-white)
- [godmode](https://godmode.space/)：需要人为每一步交互的的AutoGPT![](https://img.shields.io/badge/Auto-Agent-white)
- [agentgpt](https://agentgpt.reworkd.ai/): 基础AutoGPT![](https://img.shields.io/badge/Auto-Agent-white)
- [New Bing](https://www.bing.com/)：需要连外网否则会重定向到bing中国，需要申请waitlist ![](https://img.shields.io/badge/AIGC-Search-yellow) :star:
- [Perplexity.ai](https://www.perplexity.ai/): 同样需要科学上网，感觉比Bing做的更好的接入ChatGPT的神奇搜索引擎，在Bing之外还加入了相关推荐和追问 :star:
- [BingGPT](https://github.com/dice2o/BingGPT): NewBing开源桌面客户端，可以将聊天记录导出  ![](https://img.shields.io/badge/AIGC-Search-yellow)
- [DocsGPT](https://github.com/arc53/DocsGPT): 把ChatGPT开放域问答转化成封闭域问答的通用方案，试用垂类领域问答场景,可以试用定制的ChatBot  ![](https://img.shields.io/badge/Tool-Business-red) :star:
- [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM): 基于ChatGLM的本地知识问答，和上面的DocsGPT相似，不过可以本地部署:star:
- [ChatPDF](https://chat2doc.cn/): 国内的ChatPDF, 上传pdf后，会给出文章的Top5可能问题，然后对话式从文档中进行问答和检索，10s读3万字  ![](https://img.shields.io/badge/Tool-Business-red)
- [ChatDoc](https://chatdoc.com/?viaurl=ainavpro.com):ChatPDF升级版，增加了表格类解析，和完善的索引引用加跳转加对应文章内容高亮，哈哈我准备自己整一个 ![](https://img.shields.io/badge/Tool-Business-red)
- [ChatPaper](https://github.com/kaixindelele/ChatPaper): 根据输入关键词，自动在arxiv上下载最新的论文，并对论文进行摘要总结，可以在huggingface上试用！ ![](https://img.shields.io/badge/Tool-Business-red)
- [OpenRead](https://www.openread.academy/home): 面向论文写作，阅读场景，可以帮助生成文献综述，以及提供和NotionAI相似的智能Markdown用于写作 ![](https://img.shields.io/badge/Tool-Business-red)
- [researchgpt](https://github.com/mukulpatnaik/researchgpt): 和ChatPDF类似，支持arivx论文下载，加载后对话式获取论文重点  ![](https://img.shields.io/badge/Tool-Business-red)
- [BriefGPT](https://briefgpt.xyz/?viaurl=ainavpro.com): 日更Arxiv论文，并对论文进行摘要，关键词抽取，帮助研究者了解最新动态, UI不错哟 ![](https://img.shields.io/badge/Tool-Business-red)
- [ChatGPT-academic](https://github.com/binary-husky/chatgpt_academic): 又是一个基于gradio实现的paper润色，摘要等功能打包的实现 ![](https://img.shields.io/badge/Tool-Business-red)
- [feishu-chatgpt](https://github.com/Leizhenpeng/feishu-chatgpt): 飞书chatgpt，和365copilot相似也是多组件集成, 有点全！  ![](https://img.shields.io/badge/Tool-Business-red)
- [ChatMind](https://www.chatmind.tech/): chatgpt生成思维导图，针对话题的生成还可以，但是针对某本书的就是瞎编了，但是感觉和检索式阅读方式结合效果会出彩~  ![](https://img.shields.io/badge/Tool-Business-red)
- [Shell](): 基于ChatGPT的AI英语聊天工具，口语练习助手
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
- [Codeium](https://codeium.com/): Copilot替代品，有免费版本支持各种plugin ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [Wolverine](https://github.com/biobootloader/wolverine): 代码自我debug的python脚本 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [dreamstudio.ai](https://beta.dreamstudio.ai/dream): 开创者，Stable Difussion， 有试用quota ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F): 开创者，艺术风格为主 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [Dall.E](https://openai.com/product/dall-e-2): 三巨头这就凑齐了 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [ControlNet](https://huggingface.co/spaces/hysts/ControlNet): 为绘画创作加持可控性 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [GFPGAN](https://github.com/Nutlope/restorePhotos): 照片修复  ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [Visual ChatGPT](https://huggingface.co/spaces/microsoft/visual_chatgpt): 微软发布图像ChatGPT，对话方式进行图像生成编辑，问答 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange) :star:
- [gemo.ai](https://www.genmo.ai/): 多模态聊天机器人，包括文本，图像，视频生成

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
- [压缩即泛化，泛化即智能](https://zhuanlan.zhihu.com/p/615554635)
- [陆奇最新演讲实录：我的大模型世界观｜第十四期](https://new.qq.com/rain/a/20230423A08J7400)

## Papers
### paper List
- https://github.com/dongguanting/In-Context-Learning_PaperList
- https://github.com/thunlp/PromptPapers
- https://github.com/Timothyxxx/Chain-of-ThoughtsPapers

### Survey
- A Survey of Large Language Models
- Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing :star:
- Paradigm Shift in Natural Language Processing
- Pre-Trained Models: Past, Present and Future


### LLM Ability Analysis & Probing 
- LARGER LANGUAGE MODELS DO IN-CONTEXT LEARNING DIFFERENTLY
- Evidence of Meaning in Language Models Trained on Programs
- Sparks of Artificial General Intelligence: Early experiments with GPT-4
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

### Fix-LM Adapter Tunning
- LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS :star:
- LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning
- Parameter-Efficient Transfer Learning for NLP
- INTRINSIC DIMENSIONALITY EXPLAINS THE EFFECTIVENESS OF LANGUAGE MODEL FINE-TUNING

### Instruction Tunning LLMs 
- Flan: FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS :star:
- Flan-T5: Scaling Instruction-Finetuned Language Models
- Instruct-GPT: Training language models to follow instructions with human feedback star:
- T0: MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION
- Natural Instructions: Cross-Task Generalization via Natural Language Crowdsourcing Instructions
- Tk-INSTRUCT: SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Tasks
- Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor

### Train for Dialogue
- LaMDA: Language Models for Dialog Applications
- Sparrow: Improving alignment of dialogue agents via targeted human judgements star:
- BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage
- How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation

### Chain of Thought
- [zero-shot-COT] Large Language Models are Zero-Shot Reasoners :star:
- [Manual COT] Chain of Thought Prompting Elicits Reasoning in Large Language Models  :star:
- SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHT REASONING IN LANGUAGE MODELS
- COMPLEXITY-BASED PROMPTING FOR MULTI-STEP REASONING
- LEAST-TO-MOST PROMPTING ENABLES COMPLEX REASONING IN LARGE LANGUAGE MODELS
- Solving Quantitative Reasoning Problems with Language Models
- Specializing Smaller Language Models towards Multi-Step Reasoning
- Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters
- TEXT AND PATTERNS: FOR EFFECTIVE CHAIN OF THOUGHT IT TAKES TWO TO TANGO
- Decomposed Prompting A MODULAR APPROACH FOR Solving Complex Tasks
- Solving math word problems with processand outcome-based feedback
- CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning

### RLHF
- Deepmind
  - Teaching language models to support answers with verified quotes
  - sparrow, Improving alignment of dialogue agents via targetd human judgements :star:
- openai
  - PPO: Proximal Policy Optimization Algorithms :star:
  - Deep Reinforcement Learning for Human Preference
  - Fine-Tuning Language Models from Human Preferences
  - learning to summarize from human feedback
  - InstructGPT: Training language models to follow instructions with human feedback :star:
  - Scaling Laws for Reward Model Over optimization :star:
- Anthropic
  - A General Language Assistant as a Laboratory for Alignmen 
  - Red Teaming Language Models to Reduce Harms Methods,Scaling Behaviors and Lessons Learned
  - Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback  :star:
  - Constitutional AI Harmlessness from AI Feedback :star:
  - Pretraining Language Models with Human Preferences
- AllenAI, RL4LM：IS REINFORCEMENT LEARNING (NOT) FOR NATURAL LANGUAGE PROCESSING BENCHMARKS
- RRHF: Rank Responses to Align Language Models with Human Feedback without tears

### Agent: 让模型使用工具
- Tool Former: Toolformer: Language Models Can Teach Themselves to Use Tools
- MRKL SystemsA modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning :star:
- ReAct: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS  :star:
- Self-ask: MEASURING AND NARROWING THE COMPOSITIONALITY GAP IN LANGUAGE MODELS
- PAL: Program-aided Language Models
- HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace
- OpenAGI: When LLM Meets Domain Experts
- Tool Learning with Foundation Models

### 指令数据生成
- APE: LARGE LANGUAGE MODELS ARE HUMAN-LEVEL PROMPT ENGINEERS  :star:
- SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions :star:
- iPrompt: Explaining Data Patterns in Natural Language via Interpretable Autoprompting  
- Flipped Learning: Guess the Instruction! Flipped Learning Makes Language Models Stronger Zero-Shot Learners
- Fairness-guided Few-shot Prompting for Large Language Models  
- Instruction induction: From few examples to natural language task descriptions.
- Baize An Open-Source Chat Model with Parameter-Efficient Tuning on self-Chat Data

### 领域模型
- BioGPT：Generative Pre-trained Transformer for Biomedical Text Generation and Mining
- Galactia：A Large Language Model for Science
- PubMed GPT: A Domain-specific large language model for biomedical text
- BloombergGPT： A Large Language Model for Finance  
- ChatDoctor：Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge
- Med-PaLM：Large Language Models Encode Clinical Knowledge[V1,V2] :star:
- Augmented Large Language Models with Parametric Knowledge Guiding

### LLM Tunning Practice/Report
- BELLE: Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases
- Baize: Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data
- A Comparative Study between Full-Parameter and LoRA-based Fine-Tuning on Chinese Instruction Data for Large LM
- Exploring ChatGPT’s Ability to Rank Content: A Preliminary Study on Consistency with Human Preferences
- Towards Better Instruction Following Language Models for Chinese: Investigating the Impact of Training Data and Evaluation


### Other Prompt Engineer 
- Generated Knowledge Prompting for Commonsense Reasoning
- In-Context Instruction Learning
- PROMPTING GPT-3 TO BE RELIABLE

### Multimodal
- InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning
- Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models
- PaLM-E: An Embodied Multimodal Language Model
