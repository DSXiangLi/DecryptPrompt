# DecryptPrompt
> 如果LLM的突然到来让你感到沮丧，不妨读下主目录的Choose Your Weapon Survival Strategies for Depressed AI Academics
持续更新以下内容，Star to keep updated~

目录顺序如下
1. 国内外，垂直领域大模型
2. Agent和指令微调等训练框架
3. 开源指令，预训练，rlhf，对话，agent训练数据梳理
4. AIGC相关应用
5. prompt写作指南和5星博客等资源梳理
6. Prompt和LLM论文细分方向梳理

## My blogs
- [解密Prompt系列1. Tunning-Free Prompt：GPT2 & GPT3 & LAMA & AutoPrompt](https://cloud.tencent.com/developer/article/2215545?areaSource=&traceId=)
- [解密Prompt系列2. 冻结Prompt微调LM： T5 & PET & LM-BFF](https://cloud.tencent.com/developer/article/2223355?areaSource=&traceId=)
- [解密Prompt系列3. 冻结LM微调Prompt: Prefix-tuning & Prompt-tuning & P-tuning](https://cloud.tencent.com/developer/article/2237259?areaSource=&traceId=)
- [解密Prompt系列4. 升级Instruction Tuning：Flan/T0/InstructGPT/TKInstruct](https://cloud.tencent.com/developer/article/2245094?areaSource=&traceId=)
- [解密prompt系列5. APE+SELF=自动化指令集构建代码实现](https://cloud.tencent.com/developer/article/2260697?areaSource=&traceId=)
- [解密Prompt系列6. lora指令微调扣细节-请冷静,1个小时真不够~](https://cloud.tencent.com/developer/article/2276508)
- [解密Prompt系列7. 偏好对齐RLHF-OpenAI·DeepMind·Anthropic对比分析](https://cloud.tencent.com/developer/article/old/2289566?areaSource=&traceId=)
- [解密Prompt系列8. 无需训练让LLM支持超长输入:知识库 & Unlimiformer & PCW & NBCE ](https://cloud.tencent.com/developer/article/old/2295783?areaSource=&traceId=)
- [解密Prompt系列9. 模型复杂推理-思维链基础和进阶玩法](https://cloud.tencent.com/developer/article/old/2296079?areaSource=&traceId=)
- [解密Prompt系列10. 思维链COT原理探究](https://cloud.tencent.com/developer/article/old/2298660)
- [解密Prompt系列11. 小模型也能COT，先天不足后天补](https://cloud.tencent.com/developer/article/old/2301999)
- [解密Prompt系列12. LLM Agent零微调范式 ReAct & Self Ask](https://cloud.tencent.com/developer/article/2305421)
- [解密Prompt系列13. LLM Agent指令微调方案: Toolformer & Gorilla](https://cloud.tencent.com/developer/article/2312674)
- [解密Prompt系列14. LLM Agent之搜索应用设计：WebGPT & WebGLM & WebCPM](https://cloud.tencent.com/developer/article/2319879)
- [解密Prompt系列15. LLM Agent之数据库应用设计：DIN & C3 & SQL-Palm & BIRD](https://cloud.tencent.com/developer/article/2328749)
- [解密Prompt系列16. LLM对齐经验之数据越少越好？LTD & LIMA & AlpaGasus](https://cloud.tencent.com/developer/article/2333495)
- [解密Prompt系列17. LLM对齐方案再升级 WizardLM & BackTranslation & SELF-ALIGN](https://cloud.tencent.com/developer/article/2338592)
- [解密Prompt系列18. LLM Agent之只有智能体的世界](https://cloud.tencent.com/developer/article/2351540)
- [解密Prompt系列19. LLM Agent之数据分析领域的应用：Data-Copilot & InsightPilot](https://cloud.tencent.com/developer/article/2358413)
- [解密Prompt系列20. LLM Agent 之再谈RAG的召回多样性优化](https://cloud.tencent.com/developer/article/2365050)
- [解密Prompt系列21. LLM Agent之再谈RAG的召回信息密度和质量](https://cloud.tencent.com/developer/article/2369977)

## LLMS
### 模型评测
> 大模型评估尚未出现北极星指标，榜单排名往往和实际使用能力存在较大差异，几天没看感觉有的榜单快被玩坏了......

|榜单|结果|
|----|-----|
|[AlpacaEval：LLM-based automatic evaluation ](https://tatsu-lab.github.io/alpaca_eval/)| 开源模型王者vicuna,openchat, wizardlm|
|[Huggingface Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)|MMLU只评估开源模型，Falcon夺冠，在Eleuther AI4个评估集上评估的LLM模型榜单,vicuna夺冠| 
|[Berkley出品大模型排位赛榜有准中文榜单](https://lmsys.org/blog/2023-05-03-arena/)|Elo评分机制，GPT4自然是稳居第一，GPT4>Claude>GPT3.5>Vicuna>others|
|[CMU开源聊天机器人评测应用](https://github.com/zeno-ml/zeno-build)|ChatGPT>Vicuna>others；在对话场景中训练可能很重要|
|[Z-Bench中文真格基金评测](https://github.com/zhenbench/z-bench)|国产中文模型的编程可用性还相对较低，大家水平差不太多，两版ChatGLM提升明显|
|[Chain-of-thought评估](https://github.com/FranxYao/chain-of-thought-hub)|GSM8k, MATH等复杂问题排行榜|
|[InfoQ 大模型综合能力评估](https://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=2651170429&idx=1&sn=b98af3bd14c9f97f1aa07f0f839bb3ec&scene=21#wechat_redirect)|面向中文，ChatGPT>文心一言> Claude>星火|
|[ToolBench: 工具调用评估榜单](https://github.com/OpenBMB/ToolBench)|工具微调模型和ChatGPT进行对比，提供评测脚本|
|[AgentBench: 推理决策评估榜单](https://github.com/THUDM/AgentBench)|清华联合多高校推出不同任务环境，例如购物，家居，操作系统等场景下模型推理决策能力|
|[FlagEval](https://flageval.baai.ac.cn/#/home)|智源出品主观+客观LLM评分榜单|
|[Bird-Bench](https://bird-bench.github.io/)|更贴合真实世界应用的超大数据库，需要领域知识的NL2SQL榜单，模型追赶人类尚有时日|
|[kola](http://103.238.162.37:31622/LeaderBoard)|以世界知识为核心的评价基准，包括已知的百科知识和未知的近90天网络发布内容，评价知识记忆，理解，应用和创造能力|
|[CEVAL](https://cevalbenchmark.com/index.html#home)|中文知识评估，覆盖52个学科，机器评价主要为多项选择|
|[CMMLU](https://github.com/haonan-li/CMMLU)|67个主题中文知识和推理能力评估，多项选择机器评估|
|[LLMEval3](http://llmeval.com/)|复旦推出的知识问答榜单，涵盖大学作业和考题，题库尽可能来自非互联网避免模型作弊|

### 国外开源模型
|模型链接     | 模型描述    |
| --- | --- |
|[PPLX-7B/70B](https://blog.perplexity.ai/blog/introducing-pplx-online-llms?utm_source=labs&utm_medium=labs&utm_campaign=online-llms)|Perplexity.ai发布的自研模型，可以通过API和preplexity lab访问和RAG更搭配的LLM，在幻觉之外还对时效性推理进行了优化|
|[Mixtral](https://twitter.com/MistralAI/status/1733150512395038967)|法国“openai”开源基于MegaBlocks训练的MOE模型8*7B 32K|
|[Mistral7B](https://mistral.ai/news/announcing-mistral-7b/)|法国“openai”开源Mistral，超过llama2当前最好7B模型|
|[Dolphin-2.2.1-Mistral-7B](https://opencompass.org.cn/model-detail/Dolphin-2.2.1-Mistral-7B)|基于Mistral7B使用dolphin数据集微调|
|[LLama2](https://ai.meta.com/llama/)|Open Meta带着可商用开源的羊驼2模型来了~|
|[LLaMA](https://github.com/facebookresearch/llama)    |  Meta开源指令微调LLM，规模70 亿到 650 亿不等  |
|[WizardLM](https://github.com/nlpxucan/WizardLM)|微软新发布13B，登顶AlpacaEval开源模型Top3，使用ChatGPT对指令进行复杂度进化微调LLama2|
|[Falcon](https://huggingface.co/tiiuae/falcon-40b)   |  Falcon由阿联酋技术研究所在超高质量1万亿Token上训练得到1B，7B，40B开源，免费商用！土豪们表示钱什么的格局小了 |
|[Vicuna](https://github.com/lm-sys/FastChat)|Alpaca前成员等开源以LLama13B为基础使用ShareGPT指令微调的模型，提出了用GPT4来评测模型效果|
|[OpenChat](https://github.com/imoneoi/openchat)|80k ShareGPT对话微调LLama-2 13B开源模型中的战斗机|
|[Guanaco](https://huggingface.co/KBlueLeaf/guanaco-7B-leh)|LLama 7B基座，在alpaca52K数据上加入534K多语言指令数据微调|
|[MPT](https://huggingface.co/mosaicml/mpt-7b-chat)|MosaicML开源的预训练+指令微调的新模型，可商用，支持84k tokens超长输入|
|[RedPajama](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-3B-v1)|RedPajama项目既开源预训练数据后开源3B，7B的预训练+指令微调模型|
|[koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)|使用alpaca，HC3等开源指令集+ ShareGPT等ChatGPT数据微调llama，在榜单上排名较高|
|[ChatLLaMA](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama)     | 基于RLHF微调了LLaMA     |
|[Alpaca](https://github.com/tatsu-lab/stanford_alpaca)    |  斯坦福开源的使用52k数据在7B的LLaMA上微调得到，   |
|[Alpaca-lora](https://github.com/tloen/alpaca-lora)     |   LORA微调的LLaMA  |
|[Dromedary](https://github.com/IBM/Dromedary)|IBM self-aligned model with the LLaMA base|
|[ColossalChat](https://github.com/hpcaitech/ColossalAI)|HPC-AI Tech开源的Llama+RLHF微调|
|[MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4)|Vicuna+BLIP2 文本视觉融合|
|[StackLLama](https://huggingface.co/trl-lib/llama-7b-se-rl-peft)|LLama使用Stackexchange数据+SFT+RL|
|[Cerebras](https://huggingface.co/cerebras/Cerebras-GPT-13B)|Cerebras开源了1亿到130亿的7个模型，从预训练数据到参数全开源|
|[Dolly-v2](https://huggingface.co/databricks/dolly-v2-7b)|可商用 7b指令微调开源模型在GPT-J-6B上微调|
|[OpenChatKit](https://github.com/togethercomputer/OpenChatKit)|openai研究员打造GPT-NoX-20B微调+6B审核模型过滤|
|[MetaLM](https://github.com/microsoft/unilm)    | 微软开源的大规模自监督预训练模型    |
|[Amazon Titan](https://aws.amazon.com/cn/bedrock/titan/)|亚马逊在aws上增加自家大模型|
|[OPT-IML](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/metaseq/tree/main/projects/OPT)     |  Meta复刻GPT3，up to 175B, 不过效果并不及GPT3   |
|[Bloom](https://huggingface.co/bigscience/bloom)|BigScience出品，规模最大176B|
|[BloomZ](https://huggingface.co/bigscience/bloomz)|BigScience出品, 基于Bloom微调|
|[Galacia](https://github.com/paperswithcode/galai)|和Bloom相似，更针对科研领域训练的模型|
|[T0](https://github.com/bigscience-workshop/t-zero)|BigScience出品，3B~11B的在T5进行指令微调的模型|
|[EXLLama](https://github.com/turboderp/exllama)|Python/C++/CUDA implementation of Llama for use with 4-bit GPTQ weight|
|[LongChat](https://huggingface.co/lmsys/longchat-13b-16k)| llama-13b使用condensing rotary embedding technique微调的长文本模型|
|[MPT-30B](https://huggingface.co/mosaicml/mpt-30b)|MosaicML开源的在8Ktoken上训练的大模型|


### 国内开源模型
|模型链接     | 模型描述    |
| --- | --- |
|[Baichuan2](https://github.com/baichuan-inc/Baichuan2)|百川第二代，提供了7B/13B Base和chat的版本|
|[Baichuan](https://github.com/baichuan-inc/baichuan-7B)|百川智能开源7B大模型可商用免费|
|[ziya2](https://huggingface.co/IDEA-CCNL/Ziya2-13B-Base)|基于Llama2训练的ziya2它终于训练完了|
|[ziya](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-7B-Reward)|IDEA研究院在7B/13B llama上继续预训练+SFT+RM+PPO+HFTT+COHFT+RBRS|
|[Qwen-7B+14B+70B](https://github.com/QwenLM/Qwen-7B)|阿里开源，可商用，通义千文7B,14B,70B Base和chat模型|
|[ChatGLM3](https://github.com/THUDM/ChatGLM3)|ChatGLM3发布，支持工具调用等更多功能，不过泛化性有待评估|
|[ChatGLM2](https://github.com/thudm/chatglm2-6b)|32K长文本，FlashAttention+Multi-Query Attenion的显存优化，更强推理能力，哈哈不过很多简单问题也硬要COT，中英平行能力似乎略有下降的ChatGLM2，但是免费商用！|
|[ChatGLM](https://github.com/THUDM/ChatGLM-6B)   | 清华开源的、支持中英双语的对话语言模型，使用了代码训练，指令微调和RLHF。chatglm2支持超长文本，可免费商用啦！|
|[Yuan-2.0](https://github.com/IEIT-Yuan/Yuan-2.0)|浪潮发布Yuan2.0 2B，51B，102B|
|[YI](https://www.modelscope.cn/models/01ai/Yi-34B-Chat/summary)|元一智能开源34B，6B模型|
|[DeepSeek](https://github.com/deepseek-ai/DeepSeek-LLM)|深度求索发布的7B，67B大模型|
|[XVERSE](https://github.com/xverse-ai/XVERSE-65B)|元象发布13B免费商用大模型|
|[LLama2-chinese](https://github.com/FlagAlpha/Llama2-Chinese)|没等太久中文预训练微调后的llama2它来了~|
|[YuLan-chat2](https://github.com/RUC-GSAI/YuLan-Chat)|高瓴人工智能基于Llama-2中英双语继续预训练+指令微调/对话微调|
|[BlueLM](https://github.com/vivo-ai-lab/BlueLM)|Vivo人工智能实验室开源大模型|
|[zephyr-7B](https://ollama.ai/library/zephyr)|HuggingFace 团队基于 UltraChat 和 UltraFeedback 训练了 Zephyr-7B 模型|
|[XWin-LM](https://github.com/Xwin-LM/Xwin-LM)|llama2 + SFT + RLHF|
|[Skywork](https://github.com/SkyworkAI/Skywork)|昆仑万维集团·天工团队开源13B大模型可商用|
|[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)     |   哈工大中文指令微调的LLaMA  |
|[Moss](https://github.com/OpenLMLab/MOSS)   |  为复旦正名！开源了预训练，指令微调的全部数据和模型。可商用 |
|[InternLM](https://github.com/InternLM/InternLM)| 书生浦语在过万亿 token 数据上训练的多语千亿参数基座模型|
|[Aquila2](https://github.com/FlagAI-Open/Aquila2/blob/main/README_CN.md)|智源更新Aquila2模型系列包括全新34B|
|[Aquila](https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila)|智源开源7B大模型可商用免费|
|[UltraLM系列](https://github.com/thunlp/UltraChat)|面壁智能开源UltraLM13B，奖励模型UltraRM，和批评模型UltraCM|
|[kimi Chat](https://www.moonshot.cn/?ref=aihub.cn)|Moonshot上线超长文本LLM 可输入20W上文需要申请试用 |
|[PandaLLM](https://github.com/dandelionsllm/pandallm)|LLAMA2上中文wiki继续预训练+COIG指令微调|
|[XVERSE](https://github.com/xverse-ai/XVERSE-13B)|据说中文超越llama2的元象开源模型13B模型|
|[BiLLa](https://github.com/Neutralzz/BiLLa)|LLama词表扩充预训练+预训练和任务1比1混合SFT+指令样本SFT三阶段训练|
|[Phoenix](https://github.com/FreedomIntelligence/LLMZoo)|港中文开源凤凰和奇美拉LLM，Bloom基座，40+语言支持|
|[Wombat-7B](https://huggingface.co/GanjinZero/wombat-7b-delta)|达摩院开源无需强化学习使用RRHF对齐的语言模型, alpaca基座|
|[TigerBot](https://github.com/TigerResearch/TigerBot)|虎博开源了7B 180B的模型以及预训练和微调语料|
|[Luotuo](https://github.com/LC1332/Luotuo-Chinese-LLM)   |  中文指令微调的LLaMA，和ChatGLM   |
|[OpenBuddy](https://github.com/OpenBuddy/OpenBuddy)|Llama 多语言对话微调模型|
|[Chinese Vincuna](https://github.com/Facico/Chinese-Vicuna)|LLama 7B基座，使用Belle+Guanaco数据训练|
|[Linly](https://github.com/CVI-SZU/Linly)|Llama 7B基座，使用belle+guanaco+pclue+firefly+CSL+newscommentary等7个指令微调数据集训练|
|[Firefly](https://github.com/yangjianxin1/Firefly)| 中文2.6B模型，提升模型中文写作，古文能力，待开源全部训练代码，当前只有模型|
|[Baize](https://github.com/project-baize/baize-chatbot)    | 使用100k self-chat对话数据微调的LLama    |
|[BELLE](https://github.com/LianjiaTech/BELLE)    |使用ChatGPT生成数据对开源模型进行中文优化  |
|[Chatyuan](https://github.com/search?q=chatyuan&type=repositories)|chatgpt出来后最早的国内开源对话模型，T5架构是下面PromptCLUE的衍生模型|
|[PromptCLUE](https://github.com/clue-ai/PromptCLUE)    | 多任务Prompt语言模型    |
|[PLUG](https://www.alice-mind.com/portal#/)    |   阿里达摩院发布的大模型，提交申请会给下载链接  |
|[CPM2.0](https://baai.ac.cn/)     |  智源发布CPM2.0|
|[GLM](https://github.com/THUDM/GLM-130B) |   清华发布的中英双语130B预训练模型 |
|[BayLing](https://github.com/ictnlp/BayLing)|基于LLama7B/13B，增强的语言对齐的英语/中文大语言模型|


### 垂直领域模型&进展
|领域|模型链接     | 模型描述  
| ---| --- | --- | 
|医疗|[MedGPT](https://medgpt.co/home/zh)|医联发布的|
|医疗|MedPalm|Google在Faln-PaLM的基础上通过多种类型的医疗QA数据进行prompt-tuning指令微调得到，同时构建了MultiMedQA|
|医疗|[ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)|110K真实医患对话样本+5KChatGPT生成数据进行指令微调|
|医疗|[Huatuo](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese) [Med-ChatGLM](https://github.com/SCIR-HI/Med-ChatGLM)|医学知识图谱和chatgpt构建中文医学指令数据集+医学文献和chatgpt构建多轮问答数据|
|医疗|[Chinese-vicuna-med](https://github.com/Facico/Chinese-Vicuna/blob/master/docs/performance-medical.md)|Chinese-vicuna在cMedQA2数据上微调|
|医疗|[OpenBioMed](https://github.com/BioFM/OpenBioMed)|清华AIR开源轻量版BioMedGPT, 知识图谱&20+生物研究领域多模态预训练模型|
|医疗|[DoctorGLM](https://github.com/xionghonglin/DoctorGLM)|ChatDoctor+MedDialog+CMD 多轮对话+单轮指令样本微调GLM|
|医疗|[MedicalGPT-zh](https://github.com/MediaBrain-SJTU/MedicalGPT-zh)|自建的医学数据库ChatGPT生成QA+16个情境下SELF构建情景对话|
|医疗|[PMC-LLaMA](https://github.com/chaoyi-wu/PMC-LLaMA)|医疗论文微调Llama|
|医疗|[PULSE](https://github.com/openmedlab/PULSE)|Bloom微调+继续预训练|
|医疗|[NHS-LLM](https://github.com/CogStack/OpenGPT/tree/main)|Chatgpt生成的医疗问答，对话，微调模型|
|医疗|[神农医疗大模型](https://github.com/michael-wzhu/ShenNong-TCM-LLM)|以中医知识图谱的实体为中心生成的中医知识指令数据集11w+，微调LLama-7B|
|医疗|岐黄问道大模型|3个子模型构成，已确诊疾病的临床治疗模型+基于症状的临床诊疗模型+中医养生条理模型，看起来是要ToB落地|
|医疗|[Zhongjing](https://github.com/SupritYoung/Zhongjing)|基于Ziya-LLama+医疗预训练+SFT+RLHF的中文医学大模型|
|医疗|[MeChat](https://github.com/qiuhuachuan/smile)|心理咨询领域，通过chatgpt改写多轮对话56k|
|医疗|[SoulChat](https://github.com/scutcyr/SoulChat)|心理咨询领域中文长文本指令与多轮共情对话数据联合指令微调 ChatGLM-6B |
|医疗|[MindChat](https://github.com/X-D-Lab/MindChat)|MindChat-Baichuan-13B,Qwen-7B,MindChat-InternLM-7B使用不同基座在模型安全，共情，人类价值观对其上进行了强化|
|医疗|[DISC-MedLLM](https://github.com/FudanDISC/DISC-MedLLM)|疾病知识图谱构建QA对+QA对转化成单论对话+真实世界数据重构+人类偏好数据筛选，SFT微调baichuan|
|法律|[LawGPT-zh](https://github.com/LiuHC0428/LAW-GPT)|利用ChatGPT清洗CrimeKgAssitant数据集得到52k单轮问答+我们根据中华人民共和国法律手册上最核心的9k法律条文，利用ChatGPT联想生成具体的情景问答+知识问答使用ChatGPT基于文本构建QA对|
|法律|[LawGPT](https://github.com/pengxiao-song/LaWGPT)|基于llama+扩充词表二次预训练+基于法律条款构建QA指令微调|
|法律|[Lawyer Llama](https://github.com/AndrewZhe/lawyer-llama)|法律指令微调数据集：咨询+法律考试+对话进行指令微调|
|法律|[LexiLaw](https://github.com/CSHaitao/LexiLaw)|法律指令微调数据集：问答+书籍概念解释，法条内容进行指令微调
|法律|[ChatLaw](https://chatlaw.cloud/)|北大推出的法律大模型，应用形式很新颖类似频道内流一切功能皆融合在对话形式内|
|法律|[录问模型](https://github.com/zhihaiLLM/wisdomInterrogatory)|在baichuan基础上40G二次预训练+100K指令微调，在知识库构建上采用了Emb+意图+关键词联想结合的方案|
|金融|[FinChat.io](https://finchat.io/)|使用最新的财务数据，电话会议记录，季度和年度报告，投资书籍等进行训练|
|金融|[FinChat.io](https://finchat.io/)|使用最新的财务数据，电话会议记录，季度和年度报告，投资书籍等进行训练|
|金融|[OpenGPT](https://github.com/CogStack/OpenGPT)|领域LLM指令样本生成+微调框架|
|金融|[乾元BigBang金融2亿模型](https://github.com/ssymmetry/BBT-FinCUGE-Applications/tree/main)|金融领域预训练+任务微调|
|金融|[度小满千亿金融大模型](https://huggingface.co/xyz-nlp/XuanYuan2.0)|在Bloom-176B的基础上进行金融+中文预训练和微调|
|金融|[bondGPT](https://www.ltxtrading.com/bondgpt)|GPT4在细分债券市场的应用开放申请中|
|金融|[IndexGPT](https://www.cnbc.com/2023/05/25/jpmorgan-develops-ai-investment-advisor.html)|JPMorgan在研的生成式投资顾问|
|金融|[恒生LightGPT](https://mp.weixin.qq.com/s/vLvxvi2nOywkjt7ppiFC2g)|金融领域继续预训练+插件化设计|
|金融|[知彼阿尔法](https://finance.sina.com.cn/jjxw/2023-07-03/doc-imyzmaut2132017.shtml)|企查查商查大模型|
|金融|[AlphaBox](https://www.alphabox.top)|熵简科技发布大模型金融应用，多文档问答+会议转录+文档编辑|
|金融|[曹植](http://www.datagrand.com/products/aigc/)|达观发布金融大模型融合data2text等金融任务，赋能报告写作|
|金融|[聚宝盆](https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese)|基于 LLaMA 系基模型经过中文金融知识指令精调/指令微调(Instruct-tuning) 的微调模型|
|金融|[PIXIU](https://github.com/chancefocus/PIXIU)|整理了多个金融任务数据集加入了时间序列数据进行指令微调|
|金融|[ChatFund](https://chat.funddb.cn/)|韭圈儿发布的第一个基金大模型，看起来是做了多任务指令微调，和APP已有的数据功能进行了全方位的打通，从选基，到持仓分析等等|
|金融|[FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)|金融传统任务微调 or chatgpt生成金融工具调用|
|金融|[CFGPT](https://github.com/TongjiFinLab/CFGPT)|金融预训练+指令微调+RAG等检索任务增强|
|金融|[况客FOF智能投顾](https://pro.fofinvesting.com/workbench/home)|基金大模型应用，基金投顾，支持nl2sql类的数据查询，和基金信息对比查询等|
|金融|[DISC-FinLLM](https://github.com/FudanDISC/DISC-FinLLM)|复旦发布多微调模型组合金融系统，包括金融知识问答,金融NLP任务，金融计算，金融检索问答|
|编程|[Starcoder](https://github.com/bigcode-project/starcoder)|80种编程语言+Issue+Commit训练得到的编程大模型|
|编程|[ChatSQL](https://github.com/cubenlp/ChatSQL)|基于ChatGLM实现NL2sql|
|编程|[codegeex](http://keg.cs.tsinghua.edu.cn/codegeex/index_zh.html)|13B预训练+微调多语言变成大模型|
|编程|[codegeex2](https://github.com/THUDM/CodeGeeX2)|Chatglm2的基础上CodeGeeX2-6B 进一步经过了 600B 代码数据预训练|
|编程|[stabelcode](https://stability.ai/blog/stablecode-llm-generative-ai-coding)| 560B token多语言预训练+ 120,000 个 Alpaca指令对齐|
|编程|[SQLCoder](https://github.com/defog-ai/sqlcoder)|在StarCoder的基础上微调15B超越gpt3.5|
|数学|[MathGPT](https://www.mathgpt.com/)|是好未来自主研发的，面向全球数学爱好者和科研机构，以解题和讲题算法为核心的大模型。|
|数学|[MammoTH](https://tiger-ai-lab.github.io/MAmmoTH/)|通过COT+POT构建了MathInstruct数据集微调llama在OOD数据集上超越了WizardLM|
|数学|[MetaMath](https://github.com/meta-math/MetaMath)|模型逆向思维解决数学问题，构建了新的MetaMathQA微调llama2|
|交通|[TransGPT](https://github.com/DUOMO/TransGPT)|LLama-7B+34.6万领域预训练+5.8万条领域指令对话微调（来自文档问答）|
|交通|[TrafficGPT](https://github.com/lijlansg/TrafficGPT/tree/main)|ChatGPT+Prompt实现规划，调用交通流量领域专业TFM模型，TFM负责数据分析，任务执行，可视化等操作|
|科技|[Mozi](https://github.com/gmftbyGMFTBY/science-llm)|红睡衣预训练+论文QA数据集 + ChatGPT扩充科研对话数据|
|天文|[StarGLM](https://github.com/Yu-Yang-Li/StarGLM)|天文知识指令微调，项目进行中后期考虑天文二次预训练+KG|
|写作|[阅文-网文大模型介绍](https://www.zhihu.com/question/613058630)|签约作者内测中，主打的内容为打斗场景，剧情切换，环境描写，人设，世界观等辅助片段的生成|
|写作|[MediaGPT](https://github.com/search?q=MediaGPT&type=repositories)|LLama-7B扩充词表+指令微调，指令来自国内媒体专家给出的在新闻创作上的80个子任务|
|电商|[EcomGPT](https://github.com/Alibaba-NLP/EcomGPT)|电商领域任务指令微调大模型，指令样本250万，基座模型是Bloomz|
|评估|[Auto-J](https://modelscope.cn/models/lockonlvange/autoj-13b-fp16/summary)|上交开源了价值评估对齐13B模型|
|评估|[JudgeLM](https://github.com/baaivision/JudgeLM)|智源开源了 JudgeLM 的裁判模型，可以高效准确地评判各类大模型|
|评估|[CritiqueLLM](https://github.com/thu-coai/CritiqueLLM)|智谱AI发布评分模型CritiqueLLM,支持含参考文本/无参考文本的评估打分|

## Tool and Library
### 推理框架
| 工具描述   | 链接   | 
| --- | --- | 
|FlexFlow：模型部署推理框架|https://github.com/flexflow/FlexFlow|
|Medusa：针对采样解码的推理加速框架，可以和其他策略结合|https://github.com/FasterDecoding/Medusa|
|FlexGen: LLM推理 CPU Offload计算架构|https://github.com/FMInference/FlexGen|
|VLLM：超高速推理框架Vicuna，Arena背后的无名英雄，比HF快24倍，支持很多基座模型|https://github.com/vllm-project/vllm|
|Streamingllm: 新注意力池Attention方案，无需微调拓展模型推理长度，同时为推理提速|https://github.com/mit-han-lab/streaming-llm|
|llama2.c: llama2 纯C语言的推理框架|https://github.com/karpathy/llama2.c|

### 指令微调，预训练，rlhf框架
| 工具描述   | 链接   | 
| --- | --- | 
|LoRA：Low-Rank指令微调方案|https://github.com/tloen/alpaca-lora|
|peft：parameter-efficient prompt tunnging工具集|https://github.com/huggingface/peft|
|RL4LMs：AllenAI的RL工具|https://github.com/allenai/RL4LMs|
|RLLTE：港大，大疆等联合开源RLLTE开源学习框架|https://github.com/RLE-Foundation/rllte|
|trl：基于Transformer的强化训练框架|https://github.com/lvwerra/trl|
|trlx：分布式训练trl | https://github.com/CarperAI/trlx|
|北大开源河狸项目可复现RLHF，支持多数LLM，提供RLHF数据|https://github.com/PKU-Alignment/safe-rlhf|
|RL4LMs：AllenAI的RL工具|https://github.com/allenai/RL4LMs|
|LMFlow：港科大实验室开源的大模型微调框架，支持以上多数开源模型的指令微调和RLHF|https://github.com/OptimalScale/LMFlow|
|hugNLP:基于Huggingface开发继承Prompt技术，预训练和是指输入等多种方案|https://github.com/wjn1996/HugNLP|
|Deepspeed：针对RL训练和推理的整合优化|https://github.com/microsoft/DeepSpeed|
|Uerpy:预训练框架支持lm,mlm,unilm等|https://github.com/dbiir/UER-py|
|TecentPretrain: Uerpy的重构版本支持llama预训练|https://github.com/Tencent/TencentPretrain/tree/main|
|lamini: 整合指令数据生成，SFT，RLHF的工具库|https://github.com/lamini-ai/lamini/|
|Chain-of-thought-hub：模型推理能力评估平台|https://github.com/FranxYao/chain-of-thought-hub|
|EasyEdit：浙大开源支持多种模型，多种方案的模型知识精准编辑器|https://github.com/zjunlp/EasyEdit|
|OpenDelta：集成了各种增量微调方案的开源实现|https://github.com/thunlp/OpenDelta|
|Megablocks：MOE训练框架|https://github.com/stanford-futuredata/megablocks|
|Tutel：MOE训练框架|https://github.com/microsoft/tutel|

### Auto/Multi Agent
| 工具描述   | 链接   | 
| --- | --- | 
|AutoGen：微软开源多Agent顶层框架|https://github.com/microsoft/autogen|
|ChatDev: 面壁智能开源多智能体协作的虚拟软件公司|https://github.com/OpenBMB/ChatDev|
|Generative Agents:斯坦福AI小镇的开源代码|https://github.com/joonspk-research/generative_agents|
|BabyAGI：自执行LLM Agent|https://github.com/yoheinakajima/babyagi|
|AutoGPT：自执行LLM Agent|https://github.com/Torantulino/Auto-GPT|
|AutoGPT-Plugins：提供众多Auo-GPT官方和第三方的插件|https://github.com/Significant-Gravitas/Auto-GPT-Plugins|
|XAgent: 面壁智能开源双循环AutoGPT|https://github.com/OpenBMB/XAgent|
|MetaGPT: 覆盖软件公司全生命流程，例如产品经理等各个职业的AutoGPT|https://github.com/geekan/MetaGPT|
|ResearchGPT: 论文写作领域的AutoGPT，融合论文拆解+网络爬虫|https://github.com/assafelovic/gpt-researcher|
|MiniAGI：自执行LLM Agent|https://github.com/muellerberndt/mini-agi|
|AL Legion： 自执行LLM Agent|https://github.com/eumemic/ai-legion|
|AgentVerse：多模型交互环境 |https://github.com/OpenBMB/AgentVerse|
|AgentSims: 给定一个社会环境，评估LLM作为智能体的预定任务目标完成能力的沙盒环境|https://github.com/py499372727/AgentSims/|
|GPTRPG：RPG环境 AI Agent游戏化|https://github.com/dzoba/gptrpg|
|GPTeam：多智能体交互|https://github.com/101dotxyz/GPTeam|
|GPTEngineer：自动工具构建和代码生成|https://github.com/AntonOsika/gpt-engineer|
|WorkGPT：类似AutoGPT|https://github.com/team-openpm/workgpt|
|AI-Town: 虚拟世界模拟器|https://github.com/a16z-infra/ai-town|
|webarena:网络拟真环境，可用于自主智能体的测试，支持在线购物，论坛，代码仓库etc |https://github.com/web-arena-x/webarena|
|MiniWoB++：100+web交互操作的拟真环境 |https://github.com/Farama-Foundation/miniwob-plusplus|

### Agent工具框架类
| 工具描述   | 链接   | 
| --- | --- | 
|OpenAgents: 开源版ChatGPT-Plus搭建框架|https://github.com/xlang-ai/OpenAgents|
|langchain：LLM Agent框架|https://github.com/hwchase17/langchain|
|llama index：LLM Agent框架|https://github.com/jerryjliu/llama_index|
|Langroid: LLM Agent框架|https://github.com/langroid/langroid|
|Ragas: 评估检索增强LLM效果的框架，基于大模型prompt评估事实性，召回相关性，召回内容质量，回答相关性等|https://github.com/explodinggradients/ragas#fire-quickstart|
|fastRAG：检索框架，包括多索引检索，KG构建等基础功能|https://github.com/IntelLabs/fastRAG/tree/main|
|langflow：把langchain等agent组件做成了可拖拽式的UI|https://github.com/logspace-ai/langflow|
|Haystack: LLM Agent 框架，pipeline的设计模式个人感觉比langchain更灵活更简洁 |https://github.com/deepset-ai/haystack|
|EdgeChain: 通过Jsonnet配置文件实现LLM Agent| https://github.com/arakoodev/EdgeChains/tree/main|
|semantic-kernel：整合大模型和编程语言的SDK|https://github.com/microsoft/semantic-kernel|
|BMTTools: 清华出品多工具调用开源库，提供微调数据和评估ToolBench|https://github.com/OpenBMB/BMTools|
|Jarvis: 大模型调用小模型框架，给小模型一个未来！|https://github.com/search?q=jarvis|
|LLM-ToolMaker:让LLM自己制造Agent|https://github.com/ctlllll/LLM-ToolMaker|
|Gorilla: LLM调用大量API|https://github.com/ShishirPatil/gorilla|
|wenda:闻达小模型整合搜索用于知识融入|https://github.com/l15y/wenda|
|Alexandria: 从Arix论文开始把整个互联网变成向量索引，可以免费下载|https://alex.macrocosm.so/download|
|RapidAPI: 统一这个世界的所有API，最大API Hub，有调用成功率，latency等，是真爱！|https://rapidapi.com/hub|
|Open-Interpreter：命令行聊天框架|https://github.com/KillianLucas/open-interpreter|
|AnythingLLM: langchain推出的支持本地部署开源模型的框架|https://github.com/Mintplex-Labs/anything-llm|
|PromptFlow：微软推出的大模型应用框架|https://github.com/microsoft/promptflow|

### 其他垂直领域Agent
| 工具描述   | 链接   | 
| --- | --- | 
|Deep-KE：基于LLM对数据进行智能解析实现知识抽取|https://github.com/zjunlp/DeepKE|
|IncarnaMind：多文档RAG方案，动态chunking的方案可以借鉴|https://github.com/junruxiong/IncarnaMind|
|Vectra：平台化的LLM Agent搭建方案，从索引构建，内容召回排序，到事实检查的LLM生成|https://vectara.com/tour-vectara/|
|Data-Copilot：时间序列等结构化数据分析领域的Agent解决方案|https://github.com/zwq2018/Data-Copilot|
|DB-GPT: 以数据库为基础的GPT实验项目，使用本地化的GPT大模型与您的数据和环境进行交互|https://db-gpt.readthedocs.io/projects/db-gpt-docs-zh-cn/zh_CN/latest/index.html|
|guardrails：降低模型幻觉的python框架，promp模板+validation+修正|https://github.com/shreyar/guardrails|
|guidance：微软新开源框架，同样是降低模型幻觉的框架，prompt+chain的升级版加入逐步生成和思维链路|https://github.com/guidance-ai/guidance|
|SolidGPT: 上传个人数据，通过命令交互创建项目PRD等|https://github.com/AI-Citizen/SolidGPT|

## Training Data
| 数据类型    | 数据描述    | 数据链接    |
| --- | --- | --- |
|指令微调| self-instruct，GPT3自动生成&过滤得到指令集   |   https://github.com/yizhongw/self-instruct   |
|指令微调| Standford Alpaca：52K text-davinci-003生成的self-instruct指令数据集 |  https://github.com/tatsu-lab/stanford_alpaca   |
|指令微调| GPT4-for-LLM 中文+英文+对比指令| https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM|
|指令微调| GPTTeacher更多样的通用指令，角色扮演和代码指令| https://github.com/teknium1/GPTeacher/tree/main  |
|指令微调| 中文翻译Alpaca还有一些其他指令数据集    |    https://github.com/hikariming/alpaca_chinese_dataset https://github.com/carbonz0/alpaca-chinese-dataset|
|指令微调| alpaca指令GPT4生成，和以上几版对比显著质量更高，回复更长|https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/tree/main|
|指令微调| Guanaco数据：对Alphca指令重写后以不同语言生成总共534K，有对话和非对话类型，还有补充的QA生成样本  | https://huggingface.co/datasets/JosephusCheung/GuanacoDataset    |
|指令微调| OIG中文指令包括翻译alpaca+natural+unnatural，多轮对话，考试，leetcode指令|https://github.com/BAAI-Zlab/COIG|
|指令微调| Vicuna训练使用的样本，用API获取了sharegpt上用户和chatgpt对话历史，部分网友整理到了HF|https://github.com/domeccleston/sharegpt https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main|
|指令微调| HC3指令数据中英文，包括金融，开放QA，百科，DBQA，医学等包含人工回复|https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/tree/main|
|指令微调| MOSS开源的SFT数据包含使用plugin的对话数据|https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/tree/main|
|指令微调| InstructWild数据：用四处爬取的chatgpt指令作为种子self-instruct扩充生成，中英双语|https://github.com/XueFuzhao/InstructionWild/tree/main/data|
|指令微调| BELLE100万指令数据，参考Alpaca用ChatGPT生成，有数学，多轮对话，校色对话等等   |   https://github.com/LianjiaTech/BELLE  |
|指令微调| PromptCLUE多任务提示数据集：模板构建，只包含标准NLP任务     |  https://github.com/CLUEbenchmark/pCLUE   |
|指令微调| TK-Instruct微调用的指令数据集, 全人工标注1600+NLP任务   |   https://instructions.apps.allenai.org/  |
|指令微调| T0微调用的指令数据集（P3）    |  https://huggingface.co/datasets/bigscience/P3   |
|指令微调| p3衍生的46种多语言数据集（xmtf）    | https://github.com/bigscience-workshop/xmtf    |
|指令微调| Unnatural Instruction使用GPT3生成后改写得到240k     |   https://github.com/orhonovich/unnatural-instructions  |
|指令微调| alpaca COT对多个数据源进行了清理并统一格式放到的了HF, 重点是人工整理的COT数据|https://github.com/PhoebusSi/Alpaca-CoT|
|指令微调| 人工编写包含23种常见的中文NLP任务的指令数据，中文写作方向|https://github.com/yangjianxin1/Firefly|
|指令微调| Amazon COT指令样本包括各类QA，bigbench，math等|https://github.com/amazon-science/auto-cot|
|指令微调| CSL包含 396,209 篇中文核心期刊论文元信息 （标题、摘要、关键词、学科、门类）可做预训练可构建NLP指令任务|https://github.com/ydli-ai/CSL|
|指令微调| alpaca code 20K代码指令数据|https://github.com/sahil280114/codealpaca#data-release|
|指令微调| GPT4Tools 71K GPT4指令样本|https://github.com/StevenGrove/GPT4Tools|
|指令微调| GPT4指令+角色扮演+代码指令|https://github.com/teknium1/GPTeacher|
|指令微调| Mol-Instructions 2043K 分子+蛋白质+生物分子文本指令，覆盖分子设计、蛋白质功能预测、蛋白质设计等任务| https://github.com/zjunlp/Mol-Instructions|
|数学| 腾讯人工智能实验室发布网上爬取的数学问题APE210k|https://github.com/Chenny0808/ape210k|
|数学| 猿辅导 AI Lab开源小学应用题Math23K|https://github.com/SCNU203/Math23k/tree/main|
|数学| grade school math把OpenAI的高中数学题有改造成指令样本有2-8步推理过程|https://huggingface.co/datasets/qwedsacf/grade-school-math-instructions|
|数学| 数学问答数据集有推理过程和多项选择|https://huggingface.co/datasets/math_qa/viewer/default/test?row=2|
|数学| AMC竞赛数学题|https://huggingface.co/datasets/competition_math|
|数学| 线性代数等纯数学计算题|https://huggingface.co/datasets/math_dataset|
|代码| APPS从不同的开放访问编码网站Codeforces、Kattis 等收集的问题|https://opendatalab.org.cn/APPS|
|代码| Lyra代码由带有嵌入式 SQL 的 Python 代码组成，经过仔细注释的数据库操作程序，配有中文评论和英文评论。|https://opendatalab.org.cn/Lyra|
|代码| Conala来自StackOverflow问题,手动注释3k，英文|https://opendatalab.org.cn/CoNaLa/download|
|代码| code-alpaca ChatGPT生成20K代码指令样本|https://github.com/sahil280114/codealpaca.git|
|代码| 32K, 四种不同类型、不同难度的代码相关中文对话数据，有大模型生成，|https://github.com/zxx000728/CodeGPT|
|对话| LAION 策划的开放指令通用数据集中手动选择的组件子集 已开源40M 3万个,100M在路上 |https://github.com/LAION-AI/Open-Instruction-Generalist|
|对话| Baize基于Chat GPT构建的self-chat数据    |   https://github.com/project-baize/baize-chatbot/tree/main/data   |
|对话| FaceBook开源BlenderBot训练对话数据~6K     |   https://huggingface.co/datasets/blended_skill_talk   |
|对话| AllenAI开源38.5万个对话高质量数据集SODA   |   https://realtoxicityprompts.apps.allenai.org/ |
|对话| InstructDial在单一对话任务类型上进行指令微调   |  https://github.com/prakharguptaz/Instructdial   |
|对话| Ultra Chat 两个独立的 ChatGPT Turbo API 进行对话，从而生成多轮对话数据|https://github.com/thunlp/UltraChat|
|对话| Awesome Open-domain Dialogue Models提供多个开放域对话数据|https://github.com/cingtiye/Awesome-Open-domain-Dialogue-Models#%E4%B8%AD%E6%96%87%E5%BC%80%E6%94%BE%E5%9F%9F%E5%AF%B9%E8%AF%9D%E6%95%B0%E6%8D%AE%E9%9B%86|
|对话| Salesforce开源超全DialogStudio |https://github.com/salesforce/DialogStudio|
|对话|基于事实Reference的多轮问答中文数据，已开源5万，之后会开源更多|https://github.com/sufengniu/RefGPT|
RLFH| 北大河狸开源RLHF数据集10K，1M需要申请|https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K|
|RLHF| Anthropic hh-rlhf数据集   |   https://huggingface.co/datasets/Anthropic/hh-rlhf  |
|RLHF| Stack-exchange上问题对应多个答案，每个答案有打分|https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences/tree/main|
|RLHF| Facebook Bot Adversarial Dialogues数据集5K   |  https://github.com/facebookresearch/ParlAI  |
|RLHF| AllenAI Real Toxicity prompts   |  https://github.com/facebookresearch/ParlAI  |
|RLHF| OpenAssistant Conversations 160K消息，13500人工生成, 英文为主|https://huggingface.co/datasets/OpenAssistant/oasst1|
|RLHF| 知乎问答偏好数据集|https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k|
|RLHF| hh-rlhf中文翻译偏好数据|https://huggingface.co/datasets/liswei/rm-static-zhTW|
|RLHF|面壁智能开源大规模偏好数据，基于64Kprompt使用不同模型生成4个回答使用GPT-4评估|https://github.com/OpenBMB/UltraFeedback|
|评估集| BigBench(Beyond the Imitation Game Benchmark)    |  https://github.com/google/BIG-bench   |
|评估集| Complex QA：用于ChatGPT的评测指令集  |    https://github.com/tan92hl/Complex-Question-Answering-Evaluation-of-ChatGPT |
|评估集| Langchain开源评估数据集  |   https://huggingface.co/LangChainDatasets  |
|评估集| 2010-2022年全国高考卷的题目|https://github.com/OpenLMLab/GAOKAO-Bench|
|评估集| 中文通用大模型综合性评测基准SuperCLUE|https://github.com/CLUEbenchmark/SuperCLUE|
|英文预训练| RedPajama开源的复刻llama的预训练数据集，1.21万亿Token|https://github.com/togethercomputer/RedPajama-Data|
|英文预训练| Cerebras基于RedPajama进行清洗去重后得到的高质量数据集, 6270亿Token|https://huggingface.co/datasets/cerebras/SlimPajama-627B/tree/main/train|
|英文预训练| Pile 22个高质量数据集混合的预训练数据集800G,全量开放下载|https://pile.eleuther.ai/|
|通用预训练| UER整理CLUECorpusSmall+News Commentary中英|https://github.com/dbiir/UER-py/wiki/%E9%A2%84%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE|
|中文预训练| 智源人工智能开源的wudao 200G预训练数据|[https://github.com/BAAI-WuDao/WuDaoMM](https://openi.pcl.ac.cn/BAAI/WuDao-Data)|
|中文预训练| 里屋社区发起开源力量收集中文互联网语料集MNBVC目标是对标ChatGPT的40T|https://github.com/esbatmop/MNBVC|
|中文预训练| 复旦开源15万中文图书下载和抽取方案|https://github.com/FudanNLPLAB/CBook-150K|
|中文预训练| 书生万卷数据集来自公开网页多模态数据集，包括文本，图文和视频，其中文本1T，图文150G|https://opendatalab.org.cn/OpenDataLab/WanJuan1_dot_0|
|中文预训练| 昆仑天工开源3.2TB中英语料|https://github.com/SkyworkAI/Skywork|
|中文预训练| 浪潮开源的用于Yuan1.0训练的预训练中文语料|https://www.airyuan.cn/home|
|领域预训练| 度小满开源60G金融预训练语料|https://github.com/Duxiaoman-DI/XuanYuan|
|领域预训练| 首个中文科学文献数据集CSL,也有多种NLP任务数据 |https://github.com/ydli-ai/CSL|
|平行语料| news-commentary中英平行语料，用于中英间知识迁移|https://data.statmt.org/news-commentary/v15/training/|
|多源数据集整合| opendatalab整合了预训练阶段的多个数据源|https://opendatalab.org.cn/?industry=9821&source=JUU3JTlGJUE1JUU0JUI5JThF|
|Tool-搜索增强| webCPM开源的和搜索工具进行交互问答的数据集，包括网页抽取式摘要，多事实内容回答等人工标注数据|https://github.com/thunlp/WebCPM|
|Tool-多工具| BmTools开源的多工具调用指令数据集| https://github.com/OpenBMB/BMTools|
|Tool-多工具| AgentInstruct包含6项Agent任务，包括REACT式COT标注| https://thudm.github.io/AgentTuning/|
|Tool-多工具| MSAgent-Bench 大模型调用数据集 598k训练数据| https://modelscope.cn/datasets/damo/MSAgent-Bench/summary|
|Tool-多工具| MOSS开源的知识搜索，文生图，计算器，解方程等4个插件的30万条多轮对话数据| https://github.com/OpenLMLab/MOSS#%E6%95%B0%E6%8D%AE|
|NL2SQL|DB-GPT-Hub梳理了多源text-to-sql数据集|https://github.com/eosphoros-ai/DB-GPT-Hub|

## AIGC
- [Alpha](https://public.com/alpha?ref=supertools.therundown.ai): ChatGPT加持的金融app，支持个股信息查询，资产分析诊断，财报汇总etc 
- [Composer](https://www.composer.trade/?ref=supertools.therundown.ai)：量化策略和AI的结合，聊天式+拖拽式投资组合构建和回测
- [赛博马良](https://saibomaliang.com/):题如其名，可定制AI员工24小时全网抓取关注的创作选题，推送给小编进行二次创作  ![](https://img.shields.io/badge/Auto-Agent-white)
- [Mr.-Ranedeer-](https://github.com/JushBJJ/Mr.-Ranedeer-AI-Tutor):  基于prompt和GPT-4的强大能力提供个性化学习环境，个性化出题+模型解答
- [NexusGPT](https://nexus.snikpic.io/): AutoGPT可以出来工作了，第一个全AI Freelance平台  ![](https://img.shields.io/badge/Auto-Agent-white)
- [cognosys](https://www.cognosys.ai/create): 全网最火的web端AutoGPT，不过咋说呢试用了下感觉下巴要笑掉了，不剧透去试试你就知道 ![](https://img.shields.io/badge/Auto-Agent-white)
- [godmode](https://godmode.space/)：可以进行人为每一步交互的的AutoGPT![](https://img.shields.io/badge/Auto-Agent-white)
- [agentgpt](https://agentgpt.reworkd.ai/): 基础版AutoGPT![](https://img.shields.io/badge/Auto-Agent-white) :star:
- [do Anything](https://www.doanythingmachine.com/): AutoGPT Like的to Do List生成器  ![](https://img.shields.io/badge/Auto-Agent-white)
- [ChatMind](https://www.chatmind.tech/): chatgpt生成思维导图，模板很丰富，泛化性也不错，已经被XMind收购了~  ![](https://img.shields.io/badge/Tool-Business-red)  :star:
- [Consensus](https://consensus.app/search/): AI加持的论文搜素工具，支持多观点正负面对比 ![](https://img.shields.io/badge/AIGC-Search-yellow)
- [New Bing](https://www.bing.com/)：需要连外网否则会重定向到bing中国，需要申请waitlist ![](https://img.shields.io/badge/AIGC-Search-yellow) :star:
- [Perplexity.ai](https://www.perplexity.ai/): 同样需要科学上网，感觉比Bing做的更好的接入ChatGPT的神奇搜索引擎，在Bing之外还加入了相关推荐和追问  ![](https://img.shields.io/badge/AIGC-Search-yellow) :star:
- [devv.ai](https://devv.ai/zh/search?threadId=d5kn5g4oz2m8): 基于微调llama2 + RAG搭建的属于程序员的搜索引擎   ![](https://img.shields.io/badge/AIGC-Search-yellow)
- [BingGPT](https://github.com/dice2o/BingGPT): NewBing开源桌面客户端，可以将聊天记录导出  ![](https://img.shields.io/badge/AIGC-Search-yellow)
- [AutoLabel](https://github.com/refuel-ai/autolabel): AutoLabel标注方案 ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [Mem](https://get.mem.ai/): 笔记类产品，可以构建个人知识AI管家例如知识图谱，已获openai融资 ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [AskyourPdf](https://askyourpdf.com/zh): 同样是上传pdf进行问答和摘要的应用 ![](https://img.shields.io/badge/Tool-Business-red) 
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
- [AI Topiah](https://www.ai-topia.com/): 聆心智能AI角色聊天，和路飞唠了两句，多少有点中二之魂在燃烧 ![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [chatbase](https://www.chatbase.co/): 情感角色聊天，还没尝试 ![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [Vana](https://gptme.vana.com/login): virtual DNA, 通过聊天创建虚拟自己！概念很炫  ![](https://img.shields.io/badge/AIGC-Chatbot-blue)
- [范文喵写作](https://ai.wolian.chat/openmao/#/): 范文喵写作工具，选题，大纲，写作全流程  ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [WriteSonic](https://app.writesonic.com/)：AI写作，支持对话和定向创作如广告文案，商品描述, 支持Web检索是亮点，支持中文  ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [copy.ai](https://www.copy.ai/): WriteSonic竞品，亮点是像论文引用一样每句话都有对应网站链接，可以一键复制到右边的创作Markdown，超级好用！ ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen) :star:
- [NotionAI](https://www.notion.so/product?fredir=1)：智能Markdown，适用真相！在创作中用command调用AI辅助润色，扩写，检索内容，给创意idea ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [cal.ai](https://cal.com/ai): 自动化日程调度app
- [Quivar](https://www.quivr.app/): 一键上传各种文档，和卡片笔记进行对话的app，基于openai向量+langchain+chatgpt搭建
- [Jasper](https://www.jasper.ai/): 同上，全是竞品哈哈  ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [copy.down](https://copyai.cn/): 中文的营销文案生成，只能定向创作，支持关键词到文案的生成  ![](https://img.shields.io/badge/AIGC-AI%20wirter%20tools-brightgreen)
- [ChatExcel](https://chatexcel.com/convert): 指令控制excel计算，对熟悉excel的有些鸡肋，对不熟悉的有点用  ![](https://img.shields.io/badge/Tool-Business-red)
- [ChatPPT](https://github.com/williamfzc/chat-gpt-ppt): 使用ChatGPT进行PPT制作 ![](https://img.shields.io/badge/Tool-Business-red)
- [Eightify](https://eightify.app/zh): chrome插件，节省观看长视频的时间，立即获取关键思想，分模块总结+时间戳摘要  ![](https://img.shields.io/badge/Tool-Business-red)
- [BibiGPT](https://github.com/JimmyLv/BibiGPT): Bilibli视频内容一键总结，多模态文档  ![](https://img.shields.io/badge/Tool-Business-red)
- [AutoDev](https://ide.unitmesh.cc/):  AI编程辅助工具![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [Copilot](https://github.com/features/copilot): 要付费哟 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [Fauxpilot](https://github.com/fauxpilot/fauxpilot): copilot本地开源替代 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [CodeGex](http://codegeex.cn/zh-CN): 国内替代品，还没试过 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [Codeium](https://codeium.com/): Copilot替代品，有免费版本支持各种plugin ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [sql translate](https://www.sqltranslate.app/): text2sql，利用 OpenAI 的 API 实现的一个很简单的工具，sql到文字，文字到sql  ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [ai2sql](https://www.ai2sql.io/): text2sql老牌公司，相比sqltranslate功能更全面，支持SQL 语法检查、格式化和生成公式  ![](https://img.shields.io/badge/AIGC-Coder-blueviolet) 
- [chat2query](https://www.pingcap.com/chat2query-an-innovative-ai-powered-sql-generator-for-faster-insights/): text2sql  相比以上两位支持更自然的文本指令，以及更复杂的数据分析类的sql生成 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet) :star:
- [OuterBase](https://outerbase.com/): text2sql 设计风格很吸睛！电子表格结合mysql和dashboard，更适合数据分析宝宝 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [Chat2DB](https://github.com/chat2db/Chat2DB):智能的通用数据库SQL客户端和报表工具  ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [ChatBI](https://sf.163.com/about#event):网易数帆发布ChatBI对话数据分析平台 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [Kyligence Copilot](https://cn.kyligence.io/copilot/):Kyligence发布一站式指标平台的 AI 数智助理,支持对话式指标搜索，异动归因等等 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [Wolverine](https://github.com/biobootloader/wolverine): 代码自我debug的python脚本 ![](https://img.shields.io/badge/AIGC-Coder-blueviolet)
- [dreamstudio.ai](https://beta.dreamstudio.ai/dream): 开创者，Stable Difussion， 有试用quota ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F): 开创者，艺术风格为主 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [Dall.E](https://openai.com/product/dall-e-2): 三巨头这就凑齐了 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [ControlNet](https://huggingface.co/spaces/hysts/ControlNet): 为绘画创作加持可控性 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [GFPGAN](https://github.com/Nutlope/restorePhotos): 照片修复  ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange)
- [Visual ChatGPT](https://huggingface.co/spaces/microsoft/visual_chatgpt): 微软发布图像ChatGPT，对话方式进行图像生成编辑，问答 ![](https://img.shields.io/badge/AIGC-AI%20Artist-orange) :star:
- [gemo.ai](https://www.genmo.ai/): 多模态聊天机器人，包括文本，图像，视频生成
- [storybird](https://storybird.com/): 根据提示词生成故事绘本，还可以售卖

## Resources
### 教程类
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
- [Langchain: Chat with your data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/):吴恩达LLM实践课程
- [构筑大语言模型应用：应用开发与架构设计](https://github.com/phodal/aigc): 一本关于 LLM 在真实世界应用的开源电子书
- [Large Language Models: Application through Production](https://github.com/databricks-academy/large-language-models): 大模型应用Edx出品的课程
- [GPTSeek](https://www.gptseek.com/): 大家投票得出的最有价值的GPT应用
- [TheRunDown](https://supertools.therundown.ai/): 可订阅AI新闻的网站，也收藏了众多宝藏GPTS应用

### 书籍博客类
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
- [压缩即泛化，泛化即智能](https://zhuanlan.zhihu.com/p/615554635)  :star:
- [陆奇最新演讲实录：我的大模型世界观｜第十四期](https://new.qq.com/rain/a/20230423A08J7400)
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)  :star:
- [All You Need to Know to Build Your First LLM App](https://towardsdatascience.com/all-you-need-to-know-to-build-your-first-llm-app-eb982c78ffac)  :star:
- [GPT-4 Architecture, Infrastructure, Training Dataset, Costs, Vision, MoE](https://www.semianalysis.com/p/gpt-4-architecture-infrastructure)
- [为什么伟大不能被计划](https://weread.qq.com/web/bookDetail/93832630811e7e827g0173ca): OpenAI研究员出书
- [拾象投研机构对LLM的调研报告（文中有两次PPT的申请链接）](https://mp.weixin.qq.com/s?__biz=MjM5ODY2OTQyNg==&mid=2649769138&idx=1&sn=2c408b73f66a52e43ea991b957729519&chksm=bec3d9af89b450b95e6432dc33f4f32ae7a29cc8e2916369aad6156c5817927d1f73a0c84e82&scene=21#wechat_redirect): 对大模型应用给出了很全面的总结梳理
- [启明创投State of Generative AI 2023](https://www.guotaixia.com/post/5336.html): 最近发现应用落地才是LLM真正产生价值的核心，开始更多关注一些投研的分析报告
- [How to Use AI to Do Stuff: An Opinionated Guide](https://www.oneusefulthing.org/p/how-to-use-ai-to-do-stuff-an-opinionated)
- [Llama 2: an incredible open LLM](https://www.interconnects.ai/p/llama-2-from-meta)
- [Wolfram语言之父新书：这就是ChatGPT](https://book.douban.com/subject/36449803/?icn=index-latestbook-subject)
- [谷歌出品：对大模型领悟能力的一些探索很有意思
Do Machine Learning Models Memorize or Generalize?](https://pair.withgoogle.com/explorables/grokking/) :star:
- [OpenAI首席科学家最新讲座解读LM无监督预训练学了啥 An observation on Generalization](https://simons.berkeley.edu/talks/ilya-sutskever-openai-2023-08-14) :star:
- [The Complete Beginners Guide To Autonomous Agents](https://www.mattprd.com/p/the-complete-beginners-guide-to-autonomous-agents): Octane AI创始人 Matt Schlicht发表的关于人工智能代理的一些思考
- [An Initial Exploration of Theoretical Support for Language Model Data Engineering. Part 1: Pretraining](https://yaofu.notion.site/An-Initial-Exploration-of-Theoretical-Support-for-Language-Model-Data-Engineering-Part-1-Pretraini-dc480d9bf7ff4659afd8c9fb738086eb): 符尧大佬系列新作，通过了解大模型背后的数据工程来了解模型本质，第一篇预训练数据
- [Large Language Models (in 2023)](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650893355&idx=1&sn=5911ccc05abf5177bb71a47ea5a748c8&chksm=84e4a855b39321434a2a386c9f359979da99dd441e6cf8f88062f1e909ad8f5b9b24198d1edb&scene=0&xtrack=1) OpenAI科学家最新大模型演讲
- [OpenAI闭门会议DevDay视频 - A survey of  Techniques for Maximizing LLM performance，无法翻墙可搜标题找笔记](https://www.youtube.com/watch?v=ahnGLM-RC1Y)
- [奇绩创坛2023秋季路演日上创新LLM项目一览](https://zhuanlan.zhihu.com/p/669015906)
- [OpenAI Super Alignment Blog](https://openai.com/blog/introducing-superalignment)


## Papers
### paper List
- https://github.com/dongguanting/In-Context-Learning_PaperList
- https://github.com/thunlp/PromptPapers
- https://github.com/Timothyxxx/Chain-of-ThoughtsPapers
- https://github.com/thunlp/ToolLearningPapers
- https://github.com/MLGroupJLU/LLM-eval-survey


### 综述
- A Survey of Large Language Models
- Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing :star:
- Paradigm Shift in Natural Language Processing
- Pre-Trained Models: Past, Present and Future
- What Language Model Architecture and Pretraining objects work best for zero shot generalization  :star:
- Towards Reasoning in Large Language Models: A Survey
- Reasoning with Language Model Prompting: A Survey :star:
- An Overview on Language Models: Recent Developments and Outlook  :star:
- A Survey of Large Language Models[6.29更新版]
- Unifying Large Language Models and Knowledge Graphs: A Roadmap
- Augmented Language Models: a Survey :star:
- Domain Specialization as the Key to Make Large Language Models Disruptive: A Comprehensive Survey
- Challenges and Applications of Large Language Models
- The Rise and Potential of Large Language Model Based Agents: A Survey
- Large Language Models for Information Retrieval: A Survey
- AI Alignment: A Comprehensive Survey
- Trends in Integration of Knowledge and Large Language Models: A Survey and Taxonomy of Methods, Benchmarks, and Applications

### 大模型能力探究
- In Context Learning 
  - LARGER LANGUAGE MODELS DO IN-CONTEXT LEARNING DIFFERENTLY
  - How does in-context learning work? A framework for understanding the differences from traditional supervised learning
  - Why can GPT learn in-context? Language Model Secretly Perform Gradient Descent as Meta-Optimizers :star:
  - Rethinking the Role of Demonstrations What Makes incontext learning work? :star:
  - Trained Transformers Learn Linear Models In-Context
  - In-Context Learning Creates Task Vectors
- 涌现能力
  - Sparks of Artificial General Intelligence: Early experiments with GPT-4
  - Emerging Ability of Large Language Models :star:
  - LANGUAGE MODELS REPRESENT SPACE AND TIME
- 能力评估
  - IS CHATGPT A GENERAL-PURPOSE NATURAL LANGUAGE PROCESSING TASK SOLVER?
  - Can Large Language Models Infer Causation from Correlation?
  - Holistic Evaluation of Language Model
  - Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond
  - Theory of Mind May Have Spontaneously Emerged in Large Language Models
  - Beyond The Imitation Game: Quantifying And Extrapolating The Capabilities Of Language Models
  - Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations
  - Demystifying GPT Self-Repair for Code Generation
  - Evidence of Meaning in Language Models Trained on Programs
  - Can Explanations Be Useful for Calibrating Black Box Models
  - On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective 
  - Language acquisition: do children and language models follow similar learning stages?

### Prompt Tunning范式
- Tunning Free Prompt
  - GPT2: Language Models are Unsupervised Multitask Learners
  - GPT3: Language Models are Few-Shot Learners   :star:
  - LAMA: Language Models as Knowledge Bases?
  - AutoPrompt: Eliciting Knowledge from Language Models
- Fix-Prompt LM Tunning
  - T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
  - PET-TC(a): Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference  :star:
  - PET-TC(b): PETSGLUE It’s Not Just Size That Matters Small Language Models are also few-shot learners
  - GenPET: Few-Shot Text Generation with Natural Language Instructions
  - LM-BFF: Making Pre-trained Language Models Better Few-shot Learners  :star:
  - ADEPT: Improving and Simplifying Pattern Exploiting Training
- Fix-LM Prompt Tunning 
  - Prefix-tuning: Optimizing continuous prompts for generation  
  - Prompt-tunning: The power of scale for parameter-efficient prompt tuning :star:
  - P-tunning: GPT Understands Too :star:
  - WARP: Word-level Adversarial ReProgramming
- LM + Prompt Tunning 
  - P-tunning v2: Prompt Tuning Can Be Comparable to Fine-tunning Universally Across Scales and Tasks
  - PTR: Prompt Tuning with Rules for Text Classification
  - PADA: Example-based Prompt Learning for on-the-fly Adaptation to Unseen Domains
- Fix-LM Adapter Tunning
  - LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS :star:
  - LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning
  - Parameter-Efficient Transfer Learning for NLP
  - INTRINSIC DIMENSIONALITY EXPLAINS THE EFFECTIVENESS OF LANGUAGE MODEL FINE-TUNING

### 主流LLMS
- GLM-130B: AN OPEN BILINGUAL PRE-TRAINED MODEL
- PaLM: Scaling Language Modeling with Pathways
- PaLM 2 Technical Report
- GPT-4 Technical Report
- Backpack Language Models
- LLaMA: Open and Efficient Foundation Language Models
- Llama 2: Open Foundation and Fine-Tuned Chat Models
- Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning
- OpenBA: An Open-sourced 15B Bilingual Asymmetric seq2seq Model Pre-trained from Scratch
- Mistral 7B
- Ziya2: Data-centric Learning is All LLMs Need
- MEGABLOCKS: EFFICIENT SPARSE TRAINING WITH MIXTURE-OF-EXPERTS
- TUTEL: ADAPTIVE MIXTURE-OF-EXPERTS AT SCALE
- Phi1- Textbooks Are All You Need  :star:
- Phi1.5- Textbooks Are All You Need II: phi-1.5 technical report

###  指令微调&对齐 (instruction_tunning)
- 经典方案
   - Flan: FINETUNED LANGUAGE MODELS ARE ZERO-SHOT LEARNERS :star:
   - Flan-T5: Scaling Instruction-Finetuned Language Models
   - ExT5: Towards Extreme Multi-Task Scaling for Transfer Learning
   - Instruct-GPT: Training language models to follow instructions with human feedback :star:
   - T0: MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION
   - Natural Instructions: Cross-Task Generalization via Natural Language Crowdsourcing Instructions
   - Tk-INSTRUCT: SUPER-NATURALINSTRUCTIONS: Generalization via Declarative Instructions on 1600+ NLP Tasks
   - ZeroPrompt: Scaling Prompt-Based Pretraining to 1,000 Tasks Improves Zero-shot Generalization
   - Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor
   - INSTRUCTEVAL Towards Holistic Evaluation of Instrucion-Tuned Large Language Models
- 更少，质量更高、更多样的指令数据带来质变
    - LIMA: Less Is More for Alignment :star:
    - Maybe Only 0.5% Data is Needed: A Preliminary Exploration of Low Training Data Instruction Tuning
    - AlpaGasus: Training A Better Alpaca with Fewer Data
    - InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4
    - Instruction Mining: High-Quality Instruction Data Selection for Large Language Models
    - Visual Instruction Tuning with Polite Flamingo
    - Exploring the Impact of Instruction Data Scaling on Large Language Models:  An Empirical Study on Real-World Use Cases
- 新对齐/微调方案
   - WizardLM: Empowering Large Language Models to Follow Complex Instructions
   - Becoming self-instruct: introducing early stopping criteria for minimal instruct tuning
   - Self-Alignment with Instruction Backtranslation :star:
   - Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models
   - Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks
   - PROMPT2MODEL: Generating Deployable Models from Natural Language Instructions
   - OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs
   - Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision
   - Improving Language Model Negotiation with Self-Play and In-Context Learning from AI Feedback
   - Human-like systematic generalization through a meta-learning neural network
   - Magicoder: Source Code Is All You Need
   - Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models
- 指令数据生成
  - APE: LARGE LANGUAGE MODELS ARE HUMAN-LEVEL PROMPT ENGINEERS  :star:
  - SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions :star:
  - iPrompt: Explaining Data Patterns in Natural Language via Interpretable Autoprompting  
  - Flipped Learning: Guess the Instruction! Flipped Learning Makes Language Models Stronger Zero-Shot Learners
  - Fairness-guided Few-shot Prompting for Large Language Models  
  - Instruction induction: From few examples to natural language task descriptions .
  - SELF-QA Unsupervised Knowledge Guided alignment.
  - GPT Self-Supervision for a Better Data Annotator  
  - The Flan Collection Designing Data and Methods
  - Self-Consuming Generative Models Go MAD
  - InstructEval: Systematic Evaluation of Instruction Selection Methods
  - Overwriting Pretrained Bias with Finetuning Data
- 微调经验/实验报告
    - BELLE: Exploring the Impact of Instruction Data Scaling on Large Language Models: An Empirical Study on Real-World Use Cases
    - Baize: Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data
    - A Comparative Study between Full-Parameter and LoRA-based Fine-Tuning on Chinese Instruction Data for Large LM
    - Exploring ChatGPT’s Ability to Rank Content: A Preliminary Study on Consistency with Human Preferences
    - Towards Better Instruction Following Language Models for Chinese: Investigating the Impact of Training Data and Evaluation
- Others
   - Crosslingual Generalization through Multitask Finetuning
   - Cross-Task Generalization via Natural Language Crowdsourcing Instructions
   - UNIFIEDSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models
   - PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts
   - ROLELLM: BENCHMARKING, ELICITING, AND ENHANCING ROLE-PLAYING ABILITIES OF LARGE LANGUAGE MODELS

### 对话模型
- LaMDA: Language Models for Dialog Applications
- Sparrow: Improving alignment of dialogue agents via targeted human judgements :star:
- BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage
- How NOT To Evaluate Your Dialogue System: An Empirical Study of Unsupervised Evaluation Metrics for Dialogue Response Generation
- DialogStudio: Towards Richest and Most Diverse Unified Dataset Collection for Conversational AI
- Enhancing Chat Language Models by Scaling High-quality Instructional Conversations
- DiagGPT: An LLM-based Chatbot with Automatic Topic Management for Task-Oriented Dialogue

### 思维链 (prompt_chain_of_thought)
- 基础&进阶用法
    - [zero-shot-COT] Large Language Models are Zero-Shot Reasoners :star:
    - [few-shot COT] Chain of Thought Prompting Elicits Reasoning in Large Language Models  :star:
    - SELF-CONSISTENCY IMPROVES CHAIN OF THOUGHT REASONING IN LANGUAGE MODELS
    - LEAST-TO-MOST PROMPTING ENABLES COMPLEX REASONING IN LARGE LANGUAGE MODELS :star:
    - Tree of Thoughts: Deliberate Problem Solving with Large Language Models
    - Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models
    - Decomposed Prompting A MODULAR APPROACH FOR Solving Complex Tasks
    - Successive Prompting for Decomposing Complex Questions
    - Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework
    - Beyond Chain-of-Thought, Effective Graph-of-Thought Reasoning in Large Language Models
    - Tree-of-Mixed-Thought: Combining Fast and Slow Thinking for Multi-hop Visual Reasoning
    - LAMBADA: Backward Chaining for Automated Reasoning in Natural Language
    - Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models
    - Graph of Thoughts: Solving Elaborate Problems with Large Language Models
- 分领域COT [Math, Code, Tabular, QA]
    - Solving Quantitative Reasoning Problems with Language Models
    - SHOW YOUR WORK: SCRATCHPADS FOR INTERMEDIATE COMPUTATION WITH LANGUAGE MODELS
    - Solving math word problems with processand outcome-based feedback
    - CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning
    - T-SciQ: Teaching Multimodal Chain-of-Thought Reasoning via Large Language Model Signals for Science Question Answering
    - LEARNING PERFORMANCE-IMPROVING CODE EDITS
    - Large Language Models are Versatile Decomposers: Decompose Evidence and Questions for Table-based Reasoning 
    - Tab-CoT: Zero-shot Tabular Chain of Thought
- 原理分析
    - Towards Understanding Chain-of-Thought Prompting: An Empirical Study of What Matters  :star:
    - TEXT AND PATTERNS: FOR EFFECTIVE CHAIN OF THOUGHT IT TAKES TWO TO TANGO
    - Towards Revealing the Mystery behind Chain of Thought: a Theoretical Perspective
    - Large Language Models Can Be Easily Distracted by Irrelevant Context
- 小模型COT蒸馏
    - Specializing Smaller Language Models towards Multi-Step Reasoning   :star:
    - Teaching Small Language Models to Reason 
    - Large Language Models are Reasoning Teachers
    - Distilling Reasoning Capabilities into Smaller Language Models
    - The CoT Collection: Improving Zero-shot and Few-shot Learning of Language Models via Chain-of-Thought Fine-Tuning
- COT样本自动构建/选择
    - STaR: Self-Taught Reasoner Bootstrapping ReasoningWith Reasoning  
    - AutoCOT：AUTOMATIC CHAIN OF THOUGHT PROMPTING IN LARGE LANGUAGE MODELS
    - Large Language Models Can Self-Improve
    - Active Prompting with Chain-of-Thought for Large Language Models
    - COMPLEXITY-BASED PROMPTING FOR MULTI-STEP REASONING
- others
    - OlaGPT Empowering LLMs With Human-like Problem-Solving abilities
    - Challenging BIG-Bench tasks and whether chain-of-thought can solve them 
    - Large Language Models are Better Reasoners with Self-Verification
    - ThoughtSource A central hub for large language model reasoning data
    - Two Failures of Self-Consistency in the Multi-Step Reasoning of LLMs

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
  - The Capacity for Moral Self-Correction in Large Language Models
- AllenAI, RL4LM：IS REINFORCEMENT LEARNING (NOT) FOR NATURAL LANGUAGE PROCESSING BENCHMARKS
- 改良方案 
  - RRHF: Rank Responses to Align Language Models with Human Feedback without tears
  - PRM：Let's verify step by step 
  - Chain of Hindsight Aligns Language Models with Feedback
  - AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback
  - Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback
  - RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment
  - Training Socially Aligned Language Models in Simulated Human Society
  - RAIN: Your Language Models Can Align Themselves without Finetuning
  - Generative Judge for Evaluating Alignment
  - PEERING THROUGH PREFERENCES: UNRAVELING FEEDBACK ACQUISITION FOR ALIGNING LARGE LANGUAGE MODELS
  - SALMON: SELF-ALIGNMENT WITH PRINCIPLE-FOLLOWING REWARD MODELS
  - Large Language Model Unlearning

### LLM Agent 让模型使用工具 (llm_agent)
- 基于prompt通用方案
  - ReAct: SYNERGIZING REASONING AND ACTING IN LANGUAGE MODELS  :star:
  - Self-ask: MEASURING AND NARROWING THE COMPOSITIONALITY GAP IN LANGUAGE MODELS :star:
  - MRKL SystemsA modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning
  - PAL: Program-aided Language Models
  - ART: Automatic multi-step reasoning and tool-use for large language models
  - ReWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models  :star:
  - Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions 
  - Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models   :star:
  - Faithful Chain-of-Thought Reasoning
  - Reflexion: Language Agents with Verbal Reinforcement Learning  :star:
  - Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework
  - RestGPT: Connecting Large Language Models with Real-World RESTful APIs
  - ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based Large Language Models
  - InstructTODS: Large Language Models for End-to-End Task-Oriented Dialogue Systems
  - TPTU: Task Planning and Tool Usage of Large Language Model-based AI Agents
  - ControlLLM: Augment Language Models with Tools by Searching on Graphs
- 基于微调通用方案
  - TALM: Tool Augmented Language Models
  - Toolformer: Language Models Can Teach Themselves to Use Tools  :star:
  - Tool Learning with Foundation Models
  - Tool Maker：Large Language Models as Tool Maker
  - TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with Millions of APIs
  - AgentTuning: Enabling Generalized Agent Abilities for LLMs
- 调用模型方案
  - HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace
  - Gorilla：Large Language Model Connected with Massive APIs  :star:
  - OpenAGI: When LLM Meets Domain Experts
- 检索增强RAG
  - WebGPT：Browser-assisted question-answering with human feedback
  - WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences 
  - WebCPM: Interactive Web Search for Chinese Long-form Question Answering :star:
  - REPLUG: Retrieval-Augmented Black-Box Language Models :star:
  - Query Rewriting for Retrieval-Augmented Large Language Models
  - RETA-LLM: A Retrieval-Augmented Large Language Model Toolkit
  - Atlas: Few-shot Learning with Retrieval Augmented Language Models
  - RRAML: Reinforced Retrieval Augmented Machine Learning
  - Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation
  - PDFTriage: Question Answering over Long, Structured Documents
  - SELF-RAG: LEARNING TO RETRIEVE, GENERATE, AND CRITIQUE THROUGH SELF-REFLECTION  :star:
  - Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading  :star:
  - Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP
  - Search-in-the-Chain: Towards Accurate, Credible and Traceable Large Language Models for Knowledge-intensive 
 Tasks
  - Active Retrieval Augmented Generation
  - kNN-LM Does Not Improve Open-ended Text Generation
  - Can Retriever-Augmented Language Models Reason? The Blame Game Between the Retriever and the Language Model
  - Query2doc: Query Expansion with Large Language Models  :star:
  - RLCF：Aligning the Capabilities of Large Language Models with the Context of Information Retrieval via Contrastive Feedback
  - Augmented Embeddings for Custom Retrievals
  - DORIS-MAE: Scientific Document Retrieval using Multi-level Aspect-based Queries
  - Learning to Filter Context for Retrieval-Augmented Generation
  - THINK-ON-GRAPH: DEEP AND RESPONSIBLE REASON- ING OF LARGE LANGUAGE MODEL ON KNOWLEDGE GRAPH
  - RA-DIT: RETRIEVAL-AUGMENTED DUAL INSTRUCTION TUNING
  - Query Expansion by Prompting Large Language Models  :star:
  - CHAIN-OF-NOTE: ENHANCING ROBUSTNESS IN RETRIEVAL-AUGMENTED LANGUAGE MODELS
  - IAG: Induction-Augmented Generation Framework for Answering Reasoning Questions
  - T2Ranking: A large-scale Chinese Benchmark for Passage Ranking
  - FRESHLLMS: REFRESHING LARGE LANGUAGE MODELS WITH SEARCH ENGINE AUGMENTATION
  - ASK THE RIGHT QUESTIONS:ACTIVE QUESTION REFORMULATION WITH REINFORCEMENT LEARNING [传统方案参考]
  - Query Expansion Techniques for Information Retrieval a Survey [传统方案参考]
  - Learning to Rewrite Queries [传统方案参考]
  - Managing Diversity in Airbnb Search[传统方案参考]
- 垂直领域
  - WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
  - ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings
  - ChemCrow Augmenting large language models with chemistry tools
  - Data-Copilot: Bridging Billions of Data and Humans with Autonomous Workflow
  - Demonstration of InsightPilot: An LLM-Empowered Automated Data Exploration System
  - GeneGPT: Augmenting Large Language Models with Domain Tools for Improved Access to Biomedical Information
  - PointLLM: Empowering Large Language Models to Understand Point Clouds
  - Interpretable Long-Form Legal Question Answering with Retrieval-Augmented Large Language Models
  - Generating Explanations in Medical Question-Answering by Expectation Maximization Inference over Evidence
  - CarExpert: Leveraging Large Language Models for In-Car Conversational Question Answering
- 评估
  - Evaluating Verifiability in Generative Search Engines
  - Mind2Web: Towards a Generalist Agent for the Web
  - Auto-GPT for Online Decision Making: Benchmarks and Additional Opinions
  - API-Bank: A Benchmark for Tool-Augmented LLMs
  - ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
  - Automatic Evaluation of Attribution by Large Language Models
  - Benchmarking Large Language Models in Retrieval-Augmented Generation
  - ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems
- MultiAgent
  - Generative Agents: Interactive Simulacra of Human Behavior  :star:
  - AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors in Agents
  - CAMEL: Communicative Agents for "Mind" Exploration of Large Scale Language Model Society  :star:
  - Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf
  - Communicative Agents for Software Development  :star:
  - METAAGENTS: SIMULATING INTERACTIONS OF HUMAN BEHAVIORS FOR LLM-BASED TASK-ORIENTED COORDINATION VIA COLLABORATIVE GENERATIVE AGENTS
  - LET MODELS SPEAK CIPHERS: MULTIAGENT DEBATE THROUGH EMBEDDINGS
  - MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning
  - War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars
- 其他
  - LLM+P: Empowering Large Language Models with Optimal Planning Proficiency
  - Inference with Reference: Lossless Acceleration of Large Language Models
  - RecallM: An Architecture for Temporal Context Understanding and Question Answering
  - LLaMA Rider: Spurring Large Language Models to Explore the Open World


### LLM+KG
- 综述类
  - Unifying Large Language Models and Knowledge Graphs: A Roadmap
  - Large Language Models and Knowledge Graphs: Opportunities and Challenges
  - [知识图谱与大模型融合实践研究报告2023](https://blog.csdn.net/m0_37586850/article/details/132463508)
- KG用于大模型推理
  - Using Large Language Models for Zero-Shot Natural Language Generation from Knowledge Graphs
  - MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models
  - Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering
  - Domain Specific Question Answering Over Knowledge Graphs Using Logical Programming and Large Language Models
  - BRING YOUR OWN KG: Self-Supervised Program Synthesis for Zero-Shot KGQA
  - StructGPT: A General Framework for Large Language Model to Reason over Structured Data
- 大模型用于KG构建
  - Enhancing Knowledge Graph Construction Using Large Language Models 
  - LLM-assisted Knowledge Graph Engineering: Experiments with ChatGPT
  - ITERATIVE ZERO-SHOT LLM PROMPTING FOR KNOWLEDGE GRAPH CONSTRUCTION
  - Exploring Large Language Models for Knowledge Graph Completion

### Humanoid Agents
- HABITAT 3.0: A CO-HABITAT FOR HUMANS, AVATARS AND ROBOTS
- Humanoid Agents: Platform for Simulating Human-like Generative Agents
- Voyager: An Open-Ended Embodied Agent with Large Language Models


### 预训练数据(pretrain_data)
- DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining
- The Pile: An 800GB Dataset of Diverse Text for Language Modeling
- CCNet: Extracting High Quality Monolingual Datasets fromWeb Crawl Data
- WanJuan: A Comprehensive Multimodal Dataset for Advancing English and Chinese Large Models
- CLUECorpus2020: A Large-scale Chinese Corpus for Pre-training Language Model
- In-Context Pretraining: Language Modeling Beyond Document Boundaries

### 领域模型 (domain_llms)
- MedGPT: Medical Concept Prediction from Clinical Narratives
- BioGPT：Generative Pre-trained Transformer for Biomedical Text Generation and Mining
- Galactia：A Large Language Model for Science
- PubMed GPT: A Domain-specific large language model for biomedical text :star:
- BloombergGPT： A Large Language Model for Finance  
- ChatDoctor：Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge
- Med-PaLM：Large Language Models Encode Clinical Knowledge[V1,V2] :star:
- Augmented Large Language Models with Parametric Knowledge Guiding
- XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters
- ChatLaw Open-Source Legal Large Language Model :star:
- MediaGPT : A Large Language Model For Chinese Media
- SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support
- KITLM: Domain-Specific Knowledge InTegration into Language Models for Question Answering
- FinVis-GPT: A Multimodal Large Language Model for Financial Chart Analysis
- EcomGPT: Instruction-tuning Large Language Models with Chain-of-Task Tasks for E-commerce 
- FinGPT: Open-Source Financial Large Language Models
- TableGPT: Towards Unifying Tables, Nature Language and Commands into One GPT
- CFGPT: Chinese Financial Assistant with Large Language Model
- Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue
- LLEMMA: AN OPEN LANGUAGE MODEL FOR MATHEMATICS
- CFBenchmark: Chinese Financial Assistant Benchmark for Large Language Model

### LLM超长文本处理 (long_input)
- 位置编码、注意力机制优化
  - Unlimiformer: Long-Range Transformers with Unlimited Length Input
  - Parallel Context Windows for Large Language Models
  - [苏剑林, NBCE：使用朴素贝叶斯扩展LLM的Context处理长度](https://spaces.ac.cn/archives/9617) :star:
  - Structured Prompting: Scaling In-Context Learning to 1,000 Examples
  - Vcc: Scaling Transformers to 128K Tokens or More by Prioritizing Important Tokens
  - Scaling Transformer to 1M tokens and beyond with RMT
  - RECURRENTGPT: Interactive Generation of (Arbitrarily) Long Text 
  - TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION :star:
  - Extending Context Window of Large Language Models via Positional Interpolation
  - LongNet: Scaling Transformers to 1,000,000,000 Tokens
  - https://kaiokendev.github.io/til#extending-context-to-8k
  - [苏剑林,Transformer升级之路：10、RoPE是一种β进制编码](https://spaces.ac.cn/archives/9675) :star:
  - [苏剑林,Transformer升级之路：11、将β进制位置进行到底](https://spaces.ac.cn/archives/9706)
  - [苏剑林,Transformer升级之路：12、无限外推的ReRoPE？](https://spaces.ac.cn/archives/9708)
  - [苏剑林,Transformer升级之路：15、Key归一化助力长度外推](https://spaces.ac.cn/archives/9859)
  - EFFICIENT STREAMING LANGUAGE MODELS WITH ATTENTION SINKS
  - Ring Attention with Blockwise Transformers for Near-Infinite Context
- 上文压缩排序方案
  - Lost in the Middle: How Language Models Use Long Contexts :star:
  - LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models
  - LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression  :star:
  - Learning to Compress Prompts with Gist Tokens
  - Unlocking Context Constraints of LLMs: Enhancing Context Efficiency of LLMs with Self-Information-Based Content Filtering
- 训练方案
  - Never Lost in the Middle: Improving Large Language Models via Attention Strengthening Question Answering
  - Focused Transformer: Contrastive Training for Context Scaling
  - Effective Long-Context Scaling of Foundation Models
- 效率优化
  - HyperAttention: Long-context Attention in Near-Linear Time
  - FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

### NL2SQL
- 大模型方案
  - DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction :star:
  - C3: Zero-shot Text-to-SQL with ChatGPT  :star:
  - SQL-PALM: IMPROVED LARGE LANGUAGE MODEL ADAPTATION FOR TEXT-TO-SQL
  - BIRD Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQL :star:
  - A Case-Based Reasoning Framework for Adaptive Prompting in Cross-Domain Text-to-SQL
  - ChatDB: AUGMENTING LLMS WITH DATABASES AS THEIR SYMBOLIC MEMORY
  - A comprehensive evaluation of ChatGPT’s zero-shot Text-to-SQL capability
  - Few-shot Text-to-SQL Translation using Structure and Content Prompt Learning
- Domain Knowledge Intensive
  - Towards Knowledge-Intensive Text-to-SQL Semantic Parsing with Formulaic Knowledge
  - Bridging the Generalization Gap in Text-to-SQL Parsing with Schema Expansion
  - Towards Robustness of Text-to-SQL Models against Synonym Substitution
  - FinQA: A Dataset of Numerical Reasoning over Financial Data
- others
  - RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL
  - MIGA: A Unified Multi-task Generation Framework for Conversational Text-to-SQL

### 降低模型幻觉 (reliability)
- Survey of Hallucination in Natural Language Generation
- Trusting Your Evidence: Hallucinate Less with Context-aware Decoding  :star:
- SELF-REFINE:ITERATIVE REFINEMENT WITH SELF-FEEDBACK  :star:
- PROMPTING GPT-3 TO BE RELIABLE
- Enhancing Self-Consistency and Performance of Pre-Trained Language Models through Natural Language Inference
- On the Advance of Making Language Models Better Reasoners
- Progressive-Hint Prompting Improves Reasoning in Large Language Models
- ASK ME ANYTHING: A SIMPLE STRATEGY FOR PROMPTING LANGUAGE MODELS  :star:
- Inference-Time Intervention: Eliciting Truthful Answers from a Language Model
- Reflexion: an autonomous agent with dynamic memory and self-reflection
- Self-consistency for open-ended generations
- Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback
- Factuality Enhanced Language Models for Open-Ended Text Generation
- Adaptive Chameleon or Stubborn Sloth: Unraveling the Behavior of Large Language Models in Knowledge Clashes
- Rethinking with Retrieval: Faithful Large Language Model Inference
- RefGPT: Reference → Truthful & Customized Dialogues Generation by GPTs and for GPTs
- Enabling Large Language Models to Generate Text with Citations
- Large language models and the perils of their hallucinations
- Woodpecker: Hallucination Correction for Multimodal Large Language Models
- R-Tuning: Teaching Large Language Models to Refuse Unknown Questions
- Do Language Models Know When They’re Hallucinating References?
- LM vs LM: Detecting Factual Errors via Cross Examination
- Truth-o-meter: Collaborating with llm in fighting its hallucinations
- RARR: Researching and Revising What Language Models Say, Using Language Models
- VALIDATING LARGE LANGUAGE MODELS WITH RELM
- Language Models (Mostly) Know What They Know  :star:

### 大模型评估（evaluation）
- 事实性评估
  - TRUSTWORTHY LLMS: A SURVEY AND GUIDELINE FOR EVALUATING LARGE LANGUAGE MODELS’ ALIGNMENT
  - TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models
  - TRUE: Re-evaluating Factual Consistency Evaluation
  - SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models
  - FACTSCORE: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation
  - KoLA: Carefully Benchmarking World Knowledge of Large Language Models
  - When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories
  - FACTOOL: Factuality Detection in Generative AI A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios
- 检测任务
  - Detecting Pretraining Data from Large Language Models
  - Scalable Extraction of Training Data from (Production) Language Models
  - Rethinking Benchmark and Contamination for Language Models with Rephrased Samples


	
### 推理优化(inference)
- Fast Transformer Decoding: One Write-Head is All You Need
- Fast Inference from Transformers via Speculative Decoding
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding
- SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference
- BatchPrompt: Accomplish more with less

### 模型知识编辑黑科技(model_edit)
- ROME：Locating and Editing Factual Associations in GPT
- Transformer Feed-Forward Layers Are Key-Value Memories
- MEMIT: Mass-Editing Memory in a Transformer
- MEND：Fast Model Editing at Scale
- Editing Large Language Models: Problems, Methods, and Opportunities

### Other Prompt Engineer(prompt_engineer) 
- Calibrate Before Use: Improving Few-Shot Performance of Language Models
- In-Context Instruction Learning
- LEARNING PERFORMANCE-IMPROVING CODE EDITS
- Boosting Theory-of-Mind Performance in Large Language Models via Prompting
- Generated Knowledge Prompting for Commonsense Reasoning
- RECITATION-AUGMENTED LANGUAGE MODELS
- kNN PROMPTING: BEYOND-CONTEXT LEARNING WITH CALIBRATION-FREE NEAREST NEIGHBOR INFERENCE
- EmotionPrompt: Leveraging Psychology for Large Language Models Enhancement via Emotional Stimulus
- Causality-aware Concept Extraction based on Knowledge-guided Prompting
- LARGE LANGUAGE MODELS AS OPTIMIZERS

### Multimodal
- InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning
- Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models
- PaLM-E: An Embodied Multimodal Language Model
- LLava Visual Instruction Tuning
- MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
- TabLLM: Few-shot Classification of Tabular Data with Large Language Models
- BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions
- mPLUG-Owl : Modularization Empowers Large Language Models with Multimodality
- LVLM eHub: A Comprehensive Evaluation Benchmark for Large VisionLanguage Models
- Mirasol3B: A Multimodal Autoregressive model for time-aligned and contextual modalities

### Adversarial Attacking 
- Curiosity-driven Red-teaming for Large Language Models
- Red Teaming Language Models with Language Models
- EXPLORE, ESTABLISH, EXPLOIT: RED-TEAMING LANGUAGE MODELS FROM SCRATCH

### Others
- Pretraining on the Test Set Is All You Need 哈哈作者你是懂讽刺文学的
- Learnware: Small Models Do Big
- The economic potential of generative AI
- A PhD Student’s Perspective on Research in NLP in the Era of Very Large Language Models

