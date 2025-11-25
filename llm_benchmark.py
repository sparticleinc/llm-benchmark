import asyncio
import time
import base64
import numpy as np
from openai import AsyncOpenAI
import logging
import argparse
import json
import random
import collections
from typing import Any, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

SHORT_PROMPTS = [
    "用简单的话解释什么是人工智能。",
    "气候变化的主要成因是什么？",
    "描述植物是如何进行光合作用的。",
    "人类的免疫系统是如何运作的？",
    "第二次世界大战爆发的主要原因有哪些？",
    "用通俗的语言解释相对论。",
    "有效领导的关键原则有哪些？",
    "区块链技术是如何工作的？",
    "关于宇宙起源的主要理论有哪些？",
    "解释水循环以及它对地球生命的重要性。",
    "资本主义与社会主义有何主要区别？",
    "人类大脑是如何处理并存储记忆的？",
    "太空探索面临的主要挑战是什么？",
    "解释经济学中的供需原理。",
    "一个汉字具有左右结构，左边是木，右边是乞。这个字是什么？只需回答这个字即可。",
    "9.11 和 9.9 哪个大？",
    "描述量子计算的基本原理。",
    "人工智能如何改变医疗行业？",
    "全球变暖的主要影响是什么？",
    "可再生能源与化石燃料的区别是什么？",
    "健康饮食的基本原则是什么？",
    "心理健康的重要性是什么？",
    "工业革命如何改变了社会结构？",
    "黑洞的形成过程是什么？",
    "基因编辑技术的潜在应用有哪些？",
    "通货膨胀对经济的影响是什么？",
    "什么是供给侧经济学？",
    "文化多样性对社会的益处是什么？",
    "全球化如何影响地方文化？",
    "在线学习的优势和挑战是什么？",
    "如何提高学生的学习动机？",
]

LONG_PROMPT_PAIRS = [
    {
        "prompt": "用简单的话解释什么是人工智能。",
        "context": "人工智能 (AI) 是计算机科学中一个快速发展的领域，旨在创造能够执行通常需要人类智能的任务的智能机器。这些任务包括视觉感知、语音识别、决策和语言翻译。AI 系统被设计为能够从经验中学习，适应新的输入，并执行类似人类的任务。AI 涉及多个子领域，如机器学习、神经网络和深度学习，这些领域在自动驾驶汽车、虚拟助手和推荐系统等方面取得了显著进展。" * 109
    },
    {
        "prompt": "气候变化的主要成因是什么？",
        "context": "气候变化是一个复杂的全球现象，主要由人类活动引起，这些活动将温室气体释放到大气中。化石燃料的燃烧、森林砍伐、工业过程和农业是导致二氧化碳和其他温室气体浓度增加的主要因素。这些气体在地球周围形成一个毯子，导致地球以史无前例的速度变暖。温度模式的变化导致更多频繁和严重的天气事件、海平面上升以及全球生态系统的破坏。" * 102
    },
    {
        "prompt": "描述植物是如何进行光合作用的。",
        "context": "光合作用是一个基本的生物过程，使植物能够将光能转化为化学能。这个过程发生在植物细胞的叶绿体中，特别是在称为类囊体的结构中。叶绿素是植物中呈现绿色的色素，对捕捉光能至关重要。在光合作用中，植物通过称为气孔的小孔吸收空气中的二氧化碳，并通过根部从土壤中吸收水分。利用光能，它们将这些成分结合起来产生葡萄糖和氧气。这个过程不仅为植物提供能量，还释放出氧气，作为地球上大多数生命的副产物。" * 96
    },
    {
        "prompt": "人类的免疫系统是如何运作的？",
        "context": "人类免疫系统是由细胞、组织和器官组成的复杂网络，它们共同作用以保护身体免受有害病原体的侵害。免疫系统由两大部分组成：先天免疫系统，提供快速、非特异性的响应；以及适应性免疫系统，发展针对特定病原体的定向防御。关键成分包括白细胞（如中性粒细胞、巨噬细胞和淋巴细胞）、抗体和补体系统。免疫系统能够区分自身细胞和外来入侵者，使其能够在锁定威胁的同时尽量减少对健康组织的损害。" * 80
    },
    {
        "prompt": "第二次世界大战的主要原因是什么？",
        "context": "第二次世界大战从1939年持续到1945年，是人类历史上最致命的冲突之一。其起源可以追溯到多个复杂因素。第一次世界大战结束后的凡尔赛条约的苛刻条款使德国经济遭到破坏并心怀不满，为法西斯主义和纳粹党在阿道夫·希特勒领导下的崛起铺平了道路。纳粹德国、法西斯意大利和日本帝国的侵略性扩张政策，加上西方列强的绥靖政策，使这些政权能够不受阻碍地获得领土。战争在欧洲的直��导火索是德国于1939年9月入侵波兰，而1941年袭击珍珠港则将美国卷入冲突。" * 60
    },
    {
        "prompt": "用通俗的语言解释相对论。",
        "context": "阿尔伯特·爱因斯坦在20世纪初期发展了相对论理论，彻底改变了我们对空间、时间和引力的理解。它由两个部分组成：狭义相对论和广义相对论。狭义相对论于1905年提出，处理以非常高速度运��的物体。它提出光速对所有观察者都是恒定的，时间和空间不是绝对的，而是相对于观察者的运动。这导致了时间膨胀和长度收缩等现象。广义相对论于1915年发布，将这些思想扩展到包括引力。爱因斯坦提出，大质量物体会弯曲时空的结构，这种弯曲就是我们体验到的引力。这些理论得到了实验证据的一贯支持，并在GPS卫星等技术中有实际应用。" * 60
    },
    {
        "prompt": "有效领导的关键原则有哪些？",
        "context": "有效的领导对于引导组织、团队和个人实现目标至关重要。虽然领导风格可能有所不同，但几个关键原则被广泛认为是成功的必要条件。这些包括清晰的沟通，确保所有人理解愿景和期望；诚信，建立信任和尊重；适应���，使领导者能够在变化的环境中导航；同理心，促进强大的人际关系并理解团队动态；决策能力，能够做出及时和明智的选择；愿景，提供方向和灵感；以及赋予他人权力的能力，鼓励团队内部的成长和创新。有效的领导者还表现出对自己和团队的行动负责，并不断寻求个人成长和学习机会。" * 75
    },
    {
        "prompt": "区块链技术是如何工作的？",
        "context": "区块链是一种去中心化的分布式账本技术，是比特币等加密货币的基础，但其潜在应用远远超出数字货币。区块链的核心是一个区块链，每个区块包含一系列交易。每个区块通过加密哈希与前一个区块链接，创建一个不可更改的记录。区块链的关键创新在于其能够在一个去中心化的网络中实现共识，而无需信任任何单一实体。这通常通过工作量证明或权益证明等共识机制来实现。当发生新交易时，它会广播到一个计算机（节点）网络进行验证。一旦验证，交易就与其他交易组合创建一个新块，然后添加到链中。这个过程确保了透明度、安全性和防篡改性，使区块链适用于金融以外的各种应用，包括供应链管理、投票系统和数字身份验证。" * 60
    },
    {
        "prompt": "关于宇宙起源的主要理论有哪些？",
        "context": "宇宙起源一直是科学和哲学激烈探讨的主题。目前，最广泛接受的科学理论是大爆炸模型，该模型提出宇宙始于大约138亿年前一个无限致密和炽热的奇点，并从那时起一直在膨胀和冷却。这个理论得到的观测证据包括宇宙微波背景辐射和宇宙中轻元素的丰度。然而，关于大爆炸之前发生了什么以及它的起因仍然存在疑问。其他理论包括稳态理论，认为宇宙一直存在并在扩展时不断创造新物质，尽管由于缺乏支持证据，这一理论已不再受欢迎。更具推测性的想法包括循环宇宙的概念，其中大爆炸和大挤压在一个无限循环中发生，以及多元宇宙的想法，我们的宇宙只是许多存在的宇宙之一。" * 75
    },
    {
        "prompt": "解释水循环以及它对地球生命的重要性。",
        "context": "水循环，也称为水文循环，是水在地球和大气中连续运动的过程。它是一个复杂的系统，涉及蒸发、蒸腾、凝结、降水和径流的过程。水由于太阳能从地球表面蒸发，主要来自海洋、湖泊和河流。植物也通过蒸腾释放水蒸气。当水蒸气在大气中上升时，它冷却并凝结形成云。最终，它以雨、雪或冰雹的形式降回地球。部分水作为地表径流流过陆地，返回水体，而部分水渗入地下，补充地下水储备。这个循环对地球生命至关重要，因为它将水分布到全球各地，通过侵蚀和沉积塑造景观，调节全球温度，并提供所有生物体所需的淡水。理解和保护水循环对于管理水资源和解决气候变化和水资源匮乏等环境挑战至关重要。" * 90
    },
    {
        "prompt": "资本主义与社会主义有何主要区别？",
        "context": "资本主义和社会主义是两种对比鲜明的经济和政治体系，塑造了现代历史的大部分。资本主义的特点是生产资料的私人所有制，个人或公司拥有企业和财产。它基于自由市场竞争的原则，价格由供需决定。利润是资本主义体系中的关键动力，政府干预通常有限。相反，社会主义倡导集体或政府所有和管理生产资料和商品的分配。它旨在通过减少阶级差异并根据需要而不是支付能力分配资源来创造一个更公平的社会。在社会主义体系中，政府在经济规划和社会服务提供方面起着更大的作用。虽然纯粹形式的任何一种系统都很少见，但许多国家采用了混合经济，将资本主义和社会主义的元素结合在一起，程度不一。" * 80
    },
    {
        "prompt": "人类大脑是如何处理并存储记忆的？",
        "context": "人类大脑处理和存储记忆的能力是一个复杂而迷人的过程，涉及各种区域和神经网络。当我们经历某件事情时，感官信息首先在相关的皮层区域（如视觉皮层用于视觉，听觉皮层用于听觉）进行处理。然后，这些信息在海马体中整合，海马体是一个海马形的结构，对形成新记忆至关重要。海马体有助于将体验的不同方面结合成一个连贯的记忆，并在将短期记忆转化为长期记忆中发挥关键作用。长期记忆被认为是通过皮层广泛区域的神经元之间突触连接的变化存储的。这个过程称为巩固，可能需要数天甚至数年。不同类型的记忆（如情景记忆、语义记忆、程序性记忆）涉及不同的大脑区域和过程。记忆的检索涉及重新激活这些神经模式，这解释了为什么记忆会受到我们当前状态和环境的影响。理解这些过程对于解决记忆相关疾病和开发潜在的治疗方法至关重要。" * 90
    },
    {
        "prompt": "太空探索面临的主要挑战是什么？",
        "context": "太空探索虽然提供了巨大的科学发现和技术进步的潜力，但也面临许多挑战。一个主要障碍是太空本身的恶劣环境。太空的真空、极端温度和有害辐射对人类宇航员和敏感设备构成重大风险。长期暴露在微重力环境中会导致宇航员的健康问题，包括肌肉萎缩和骨密度下降。后勤挑战也很大：太空旅行中涉及的巨大距离需要先进的推进系统和仔细的资源管理。将有效载荷发射到轨道仍然非常昂贵，限制了任务的范围和频率。通信延迟对深空任务来说越来越成问题，需要航天器和探测器具有高度的自主性。此外，地球轨道上的太空碎片对卫星和航天器构成日益严重的威胁。随着我们展望在月球或火星上建立基地等长期目标，我们面临着在创建可持续栖息地以及在长期任务中对船员成员的心理影响进行管理方面的新挑战。尽管存在这些障碍，持续的研究和技术创新继续推动太空探索的可能性边界。" * 80
    },
    {
        "prompt": "解释经济学中的供需原理。",
        "context": "供需是经济学中的一个基本概念，描述了市场中商品或服务的价格和数量是如何通过买卖双方的互动决定的。需求定律指出，在其他条件相同的情况下，随着产品价格的上涨，消费者需求的数量减少。这通常由向下倾斜的需求曲线表示。相反，供给定律指出，随着产品价格的上涨，生产者愿意供给的数量增加，由向上倾斜的供给曲线表示。这两条曲线交叉的点称为均衡点，决定了市场价格和数量。这个模型有助于解释价格如何随着供需变化而波动。例如，如果需求增加而供给保持不变，价格将上涨。如果供给增加而需求保持不变，价格将下降。理解供需对于分析市场行为、预测价格变化和制定经济政策至关重要。" * 70
    },
    {
        "prompt": "民主政府的主要特征是什么？",
        "context": "民主政府是一种以人民统治原则为基础的治理系统。虽然民主可以采取多种形式，但它们通常共享几个关键特征。首先是自由和公正的选举概念，公民有权在定期的间隔投票选举他们的代表。这与政治多元化原则密切相关，允许多个政党和观点竞争权力。保护个人权利和公民自由，如言论自由、新闻自由和集会自由，是民主的另一个关键方面。权力分立通常被实施以防止权力集中，通常将政府分为行政、立法和司法部门，相互提供制衡。法治，确保所有公民，包括掌权者，均受法律的约束，是民主治理的基础。政府运作的透明度和问责制，通常通过自由媒体和积极的公民社会来促进，帮助维护民主原则。此外，许多民主强调少数群体权利的保护和多数统治与少数权利的概念，旨在平衡多数的意愿与所有公民的基本权利。" * 85
    },
    {
        "prompt": "疫苗如何预防疾病？",
        "context": "疫苗是预防传染病的最有效工具之一，通过利用身体自身的免疫系统起作用。当病原体如病毒或细菌进入体内时，免疫系统通过产生特定于该病原体的抗体来响应。这些抗体有助于中和或摧毁入侵者。疫苗通过将病原体的无害形式引入体内——无论是减弱的、灭活的还是仅仅是部分病原体——模仿这一自然过程。这刺激免疫系统产生针对该病原体的抗体和记忆细胞，而不会引起实际疾病。如果接种疫苗的人后来遇到真正的病原体，他们的免疫系统可以迅速识别并迅速有效地响应，通常完全防止疾病或减少其严重性。有些疫苗需要多次接种或定期加强来维持免疫力。群体免疫的概念在疫苗接种策略中也很重要：当大部分人口接种疫苗时，病原体的传播变得困难，从而间接保护那些无法接种疫苗的人。疫苗技术的进步，如 mRNA 疫苗，正在扩展我们快速开发针对新威胁的疫苗的能力。" * 80
    },
    {
        "prompt": "人类进化的主要理论是什么？",
        "context": "人类进化是研究我们物种智人及其祖先的生物和文化发展的学科。解释人类进化的主要科学理论基于达尔文的自然选择进化理论，已适应并包含现代基因学理解。该理论提出人类从早期灵长类物种进化而来，历时数百万年。关键思想包括共同祖先的概念，暗示人类与其他灵长类动物，特别是类人猿共享共同祖先。“走出非洲”理论指出，现代人类起源于非洲，然后迁徙到世界其他地方。化石证据揭示了一系列中间物种，如南方古猿、能人和直立人，显示了在脑容量、双足行走和工具使用等特征上的逐渐变化。最近的发现和基因研究使这一图景复杂化，表明不同人类物种（如智人和尼安德特人）之间的杂交以及可能的多次非洲迁徙。古生物学、遗传学和考古学的持续研究不断完善我们对人类进化的理解，常常挑战先前的假设并揭示我们物种复杂的历史。" * 75
    },
    {
        "prompt": "描述板块构造过程及其对地球的影响。",
        "context": "板块构造是地质学中的一个基本理论，用来解释地球岩石圈的大规模运动。该理论提出地球的外层被分为几个大的刚性板块，这些板块相互移动。这些板块漂浮在半流体的软流圈上，并由地幔中的对流驱动。板块边界分为三种类型：离散边界，板块在此分开并形成新的地壳；汇聚边界，板块在此碰撞，导致俯冲或造山；和转换边界，板块在此水平滑动。板块构造过程对地球表面和内部结构有深远的影响。它是山脉、海洋盆地和岛弧形成的原因。它还在岩石循环、火山活动和地震发生中起着至关重要的作用。从地质时期来看，板块构造影响了气候模式、洋流以及全球动植物的分布。理解板块构造对预测地质灾害、解释自然资源分布以及理解地球的长期地质历史至关重要。" * 70
    },
    {
        "prompt": "生物多样性丧失的主要原因是什么？",
        "context": "生物多样性丧失，即地球上生命形式的多样性下降，是一个具有深远影响的关键环境问题，影响生态系统和人类福祉。几个相互关联的因素导致了这种丧失。栖息地破坏和碎片化，通常由人类活动如森林砍伐、城市化和农业扩张引起，是主要驱动因素。气候变化日益被认为是一个主要威胁，改变生态系统的速度超过许多物种的适应能力。自然资源的过度开发，包括过度捕捞和偷猎，直接减少了许多物种的种群数量。污染，包括化学径流、塑料废物和空气污染，破坏栖息地并危害野生动物。入侵物种的引入，通常是由于人类活动，可能破坏当地生态系统并与本地物种竞争。此外，疾病的传播，有时由于气候变化和栖息地压力而加剧，可能对某些物种的种群造成毁灭性影响。这些因素常常相互作用并加剧彼此的影响，加速了生物多样性丧失的速度。解决这一危机需要全面的保护策略、可持续的资源管理和全球合作以减轻人类对自然生态系统的影响。" * 75
    },
    {
        "prompt": "描述量子计算的基本原理。",
        "context": "量子计算是一种利用量子力学现象（如叠加和纠缠）来执行计算的技术。传统计算机使用位作为信息的基本单位，每个位可以是0或1。相比之下，量子计算机使用量子比特或'量子位'，它们可以同时处于0和1的叠加状态。这种特性使量子计算机能够并行处理大量可能的计算路径，理论上可以解决某些传统计算机难以解决的问题。量子计算的应用包括密码学、药物发现、优化问题和材料科学等领域。尽管量子计算有巨大潜力，但它仍面临着量子退相干和错误校正等技术挑战。" * 137
    },
    {
        "prompt": "人工智能如何改变医疗行业？",
        "context": "人工智能正在深刻改变医疗行业，提高诊断准确性、治疗效果和患者护理。在诊断方面，AI算法能够分析医学图像（如X光片、CT扫描和MRI），有时比人类医生更快、更准确地识别疾病模式。预测分析工具利用患者数据预测健康风险和疾病进展，使医疗专业人员能够采取预防措施。在药物发现中，AI加速了新药的开发过程，分析大量化合物数据以识别潜在的治疗方法。机器人辅助手术提高了精确度和减少了恢复时间。虚拟健康助手和聊天机器人改善了患者参与度和初级护理的可及性。尽管AI在医疗中有巨大潜力，但仍存在数据隐私、算法偏见和监管挑战等问题需要解决。" * 109
    },
    {
        "prompt": "全球变暖的主要影响是什么？",
        "context": "全球变暖正在对地球的物理、生物和人类系统产生广泛的影响。气温上升导致极端天气事件更加频繁和严重，包括热浪、干旱、野火和强烈风暴。冰川和极地冰盖的融化正在导致海平面上升，威胁着沿海社区和低洼岛屿。海洋酸化和珊瑚白化正在破坏海洋生态系统。陆地上，物种正在改变其地理分布，有些面临灭绝风险，因为它们无法适应变化的条件。对人类社会而言，全球变暖威胁着粮食和水安全，加剧了健康风险，如热相关疾病和传染病的传播范围扩大。气候变化还可能加剧社会不平等和引发大规模移民，因为人们逃离受影响最严重的地区。" * 118
    },
    {
        "prompt": "可再生能源与化石燃料的区别是什么？",
        "context": "可再生能源和化石燃料在来源、环境影响和可持续性方面有根本区别。可再生能源来自自然过程中不断补充的来源，如阳光、风、水流、地热热量和生物质。相比之下，化石燃料（煤炭、石油和天然气）是从地下提取的有限资源，形成于数百万年前的古代植物和生物残骸。环境影响方面，可再生能源通常产生很少或没有温室气体排放，而化石燃料燃烧是气候变化的主要贡献者。可再生能源技术的初始安装成本可能较高，但运营成本通常较低，而且随着技术进步，成本持续下降。化石燃料的提取和使用可能导致空气和水污染、栖息地破坏和公共健康问题。从长远来看，可再生能源提供了一条更可持续的能源路径，而化石燃料最终将耗尽。" * 110
    },
    {
        "prompt": "健康饮食的基本原则是什么？",
        "context": "健康饮食的基本原则围绕着平衡、多样性和适度。均衡的饮食应包括所有主要食物组的适当比例：水果和蔬菜、全谷物、蛋白质来源和健康脂肪。多样性确保摄入广泛的营养素，包括必需维生素、矿物质和抗氧化剂。适量控制总热量摄入对维持健康体重至关重要。减少加工食品、添加糖、反式脂肪和过量钠的摄入可以降低慢性疾病风险。增加膳食纤维摄入支持消化健康，而充足的水分摄入对整体健康至关重要。重要的是要认识到个体差异，如年龄、性别、活动水平和健康状况会影响特定的营养需求。可持续的饮食模式，如地中海饮食，强调以植物为基础的食物、适量的鱼类和减少红肉，已被证明对长期健康有益。" * 118
    },
    {
        "prompt": "心理健康的重要性是什么？",
        "context": "心理健康是整体健康的基本组成部分，与身体健康同等重要。良好的心理健康使个体能够应对生活的正常压力、发挥生产力、实现潜力并为社区做出贡献。它影响我们的思考、感受和行为方式，塑造我们如何做决定、处理压力和与他人互动。心理健康问题，如抑郁症和焦虑症，可以显著影响日常功能、人际关系和生活质量。它们还可能导致或加剧身体健康问题，如心脏病、中风和糖尿病。尽管心理健康问题很常见，影响全球近10亿人，但污名和误解仍然阻碍许多人寻求帮助。早期干预和适当治疗可以有效管理大多数心理健康状况。促进心理健康的策略包括培养社会联系、保持身体活动、发展应对技能、获得充足睡眠和寻求专业支持。" * 127
    },
    {
        "prompt": "工业革命如何改变了社会结构？",
        "context": "工业革命从18世纪末到19世纪初彻底改变了社会结构，将主要是农业的社会转变为以制造业和工业为中心的社会。这一转变催生了城市化的快速增长，农村人口涌入城市寻找工厂工作，导致城市扩张和新的城市中心的发展。新兴的工厂系统创造了一个新的工人阶级，同时也扩大了中产阶级，包括工厂主、商人和专业人士。这导致了阶级结构的重组和新的社会分层。工作性质发生了根本变化，从以家庭为基础的手工艺和农业转向工厂的机械化生产。这影响了家庭结构，因为工作与家庭生活分离，性别角色转变。工业革命还推动了教育的扩展，因为识字和技术技能变得越来越重要。政治上，工人阶级的崛起最终导致了工会运动、劳工改革和扩大政治代表权的呼声。技术进步改变了日常生活的各个方面，从交通到通信，创造了更加互联互通的社会。" * 110
    },
    {
        "prompt": "黑洞的形成过程是什么？",
        "context": "黑洞是时空中引力极强的区域，甚至连光都无法逃脱。它们主要通过大质量恒星的死亡形成。当一颗至少太阳质量约20倍的恒星耗尽其核燃料时，核心开始坍塌。这种坍塌产生的压力通常会导致超新星爆发，但对于特别大的恒星，即使是这种爆发也无法抵抗引力的内向拉力。结果，物质被压缩成一个无限密度的奇点，周围是事件视界——一个一旦越过就无法返回的边界。黑洞的大小由其质量决定，较大的黑洞有较大的事件视界。超大质量黑洞，质量为太阳的数百万到数十亿倍，被认为存在于大多数星系的中心，包括我们的银河系。这些可能是通过较小黑洞的合并和吸积大量物质而形成的。最近的研究表明，原初黑洞可能在宇宙早期形成，为我们对这些神秘天体的理解增添了新的维度。" * 110
    },
    {
        "prompt": "基因编辑技术的潜在应用有哪些？",
        "context": "基因编辑技术，特别是CRISPR-Cas9，正在彻底改变我们操纵DNA的能力，开辟了广泛的潜在应用。在医学领域，基因编辑有望治疗遗传疾病，如囊性纤维化、镰状细胞贫血和亨廷顿舞蹈症，通过直接修正导致这些疾病的基因突变。它还在癌症研究中显示出前景，使科学家能够设计免疫细胞更有效地靶向肿瘤。在农业中，基因编辑可以开发抗病、抗旱和营养价值更高的作物品种，潜在地提高粮食安全和可持续性。在生物技术领域，它使研究人员能够设计微生物生产药物、生物燃料和其他有价值的化合物。尽管基因编辑有巨大潜力，但它也引发了伦理问题，特别是关于人类胚胎编辑、基因驱动生物体的环境影响以及获取这些技术的公平性问题。" * 127
    },
    {
        "prompt": "通货膨胀对经济的影响是什么？",
        "context": "通货膨胀，即一段时间内物价总体水平的上升，对经济有广泛的影响。适度的通货膨胀（通常为2-3%）通常被认为是健康的，因为它鼓励消费和投资，因为人们倾向于现在购买而不是等待，并促进经济增长。然而，高通胀或不可预测的通胀可能具有破坏性。它侵蚀了购买力，特别是对固定收入者，如退休人员。它创造了不确定性，使企业难以规划和投资。通货膨胀扭曲了资源分配，因为人们可能转向投机性资产而不是生产性投资。它还可能导致'菜单成本'——企业频繁更新价格的费用。在国际上，通货膨胀影响汇率和贸易竞争力。中央银行通常通过调整利率来控制通货膨胀，提高利率以减缓经济并降低通胀压力。政府可能还实施财政政策，如减少支出或增加税收，以应对通货膨胀。" * 110
    },
    {
        "prompt": "什么是供给侧经济学？",
        "context": "供给侧经济学是一种经济理论，强调通过减少税收和减少监管来刺激经济增长。这一理论在1970年代和1980年代由经济学家如罗伯特·蒙代尔、阿瑟·拉弗和保罗·克雷格·罗伯茨发展，并与里根政府的政策密切相关。供给侧经济学的核心是拉弗曲线的概念，它表明降低税率可以增加政府收入，因为它鼓励更多的经济活动。支持者认为，降低富人和企业的税收将导致更多的投资、就业创造和整体经济扩张，最终使所有收入水平的人受益——一种被称为'涓滴经济学'的效应。批评者认为这些政策主要使富人受益，导致收入不平等加剧，并可能导致预算赤字。供给侧经济学与凯恩斯经济学形成对比，后者强调需求在驱动经济增长中的作用，并倡导政府支出来刺激经济。" * 118
    },
    {
        "prompt": "文化多样性对社会的益处是什么？",
        "context": "文化多样性为社会带来了许多益处，丰富了集体经验并增强了社会弹性。多元文化社会往往促进创新和创造力，因为不同背景的人带来各种观点和解决问题的方法。这种多样性的思维方式在商业环境中特别有价值，研究表明多元化团队通常更具创新性和生产力。文化交流促进了相互理解和宽容，减少了偏见和刻板印象，并有助于建立更具包容性的社会。多样性还增强了教育体验，使学生接触到不同的世界观和历史视角。在全球化经济中，文化多样性提供了竞争优势，使社会能够更有效地与不同的市场和客户群体互动。它还丰富了社会的文化生活，通过各种艺术、音乐、美食和传统。最后，多元文化社会往往更具适应性和弹性，能够从多种知识和经验中汲取应对挑战。" * 118
    },
    {
        "prompt": "全球化如何影响地方文化？",
        "context": "全球化对地方文化产生了深远而复杂的影响，带来了机遇和挑战。一方面，全球化促进了文化交流，使地方传统和做法能够在国际上得到认可和欣赏。数字技术和社交媒体使边缘化社区能够保存和分享他们的文化遗产。全球市场为地方工匠和艺术家提供了新的经济机会。另一方面，全球化可能导致文化同质化，随着西方和主流文化的传播，地方传统被边缘化。消费主义和商业化可能使文化表达变得商品化，削弱其真实性和意义。语言多样性尤其受到威胁，许多少数民族语言面临消失的风险。对这些影响的反应各不相同，从文化保护主义到'全球本地化'——全球影响与地方特色的融合，创造出独特的混合形式。最终，全球化对地方文化的影响取决于社区如何在保持其独特身份的同时适应和参与更广泛的全球对话。" * 110
    },
    {
        "prompt": "在线学习的优势和挑战是什么？",
        "context": "在线学习提供了显著的优势，同时也带来了独特的挑战。在优势方面，它提供了前所未有的灵活性，使学习者能够按照自己的节奏学习并围绕现有承诺安排学习。它消除了地理障碍，使世界各地的人们能够获得以前可能无法获得的教育机会。在线课程通常比传统教育更经济实惠，减少了与通勤、住宿和实体材料相关的成本。数字平台支持各种学习风格，通过视频、交互式练习和讨论论坛提供内容。然而，挑战也很重要。许多学生在缺乏面对面互动和即时反馈的情况下挣扎。自律和时间管理成为成功的关键因素。技术问题和数字鸿沟可能限制某些人群的获取。在线环境可能缺乏传统课堂的社交和协作方面。某些学科，特别是那些需要实践经验的学科，可能难以有效地在线教授。尽管存在这些挑战，在线学习的持续发展和混合模式的出现表明，它将继续成为教育格局的重要组成部分。" * 102
    },
    {
        "prompt": "如何提高学生的学习动机？",
        "context": "提高学生的学习动机涉及多种策略，针对学习过程的认知和情感方面。创造相关性至关重要——当学生理解材料如何与他们的生活、目标和兴趣相关时，他们更有可能投入。提供自主权和选择让学生对自己的学习有所有权，增强内在动机。设定明确、可实现但具有挑战性的目标帮助学生看到进步并保持专注。及时、具体和建设性的反馈支持能力感和持续改进。培养成长心态，强调努力和策略而不是固定能力，鼓励学生面对挑战时的坚韧。创造支持性的学习环境，学生感到安全、受尊重并与同伴和教育者有联系，满足了归属感的基本心理需求。使用多样化的教学方法，包括协作学习、基于问题的活动和技术整合，可以吸引不同的学习风格和偏好。认可和庆祝成就，无论大小，都能增强自信并激励持续努力。最后，教育者和家长建模的热情和终身学习的价值对塑造学生对学习的态度有强大影响。" * 96
    }
]


def _normalize_api_key(api_key: Optional[str]) -> Optional[str]:
    """将占位 api_key 转换为 None，便于后续逻辑判断。"""
    if not api_key:
        return None
    if api_key.strip().lower() == "default":
        return None
    return api_key


def _build_basic_auth_header(api_key: Optional[str], auth_config: Dict[str, Any]) -> Optional[str]:
    """根据现有信息构造 Basic Auth Header。"""
    if api_key and api_key.lower().startswith("basic "):
        return api_key.strip()

    username = auth_config.get("basic_auth_user")
    password = auth_config.get("basic_auth_password")
    if username is not None and password is not None:
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")
        return f"Basic {token}"

    return None


def _create_llm_client(llm_url: str, api_key: Optional[str], auth_config: Optional[Dict[str, Any]]) -> AsyncOpenAI:
    """根据认证配置创建 AsyncOpenAI 客户端，支持 Bearer、Basic 以及无认证。"""
    auth_config = auth_config or {}
    auth_header_override = auth_config.get("auth_header")

    if auth_header_override:
        return AsyncOpenAI(base_url=llm_url, api_key="", default_headers={"Authorization": auth_header_override})

    normalized_api_key = _normalize_api_key(api_key)
    requested_type = (auth_config.get("auth_type") or "auto").lower()
    valid_types = {"auto", "bearer", "basic", "none"}
    if requested_type not in valid_types:
        raise ValueError(f"Unsupported auth_type '{requested_type}'. Valid options: {', '.join(sorted(valid_types))}")

    if requested_type == "auto":
        if auth_config.get("basic_auth_user") is not None and auth_config.get("basic_auth_password") is not None:
            auth_type = "basic"
        elif normalized_api_key and normalized_api_key.lower().startswith("basic "):
            auth_type = "basic"
        elif normalized_api_key:
            auth_type = "bearer"
        else:
            auth_type = "none"
    else:
        auth_type = requested_type

    default_headers = None
    client_api_key: Optional[str] = normalized_api_key

    if auth_type == "basic":
        header_value = _build_basic_auth_header(normalized_api_key, auth_config)
        if not header_value:
            raise ValueError(
                "Basic auth requires an API key starting with 'Basic ', a fully specified --auth_header, "
                "or both --basic_auth_user/--basic_auth_password."
            )
        default_headers = {"Authorization": header_value}
        client_api_key = ""
    elif auth_type == "none":
        client_api_key = ""
    elif auth_type != "bearer":
        raise ValueError(f"Unsupported auth_type '{auth_type}'.")

    if client_api_key is None:
        # Bearer 模式下必须明确提供 key；如果没有，尽早报错
        raise ValueError("Bearer authentication requires a valid --api_key or OPENAI_API_KEY.")

    # 确保URL以/v1结尾，这对OpenAI兼容API很重要
    if not llm_url.endswith('/v1'):
        if llm_url.endswith('/'):
            base_url = llm_url + 'v1'
        else:
            base_url = llm_url + '/v1'
    else:
        base_url = llm_url

    logging.info(f"使用认证类型: {auth_type}")
    logging.info(f"API端点: {base_url}")

    return AsyncOpenAI(base_url=base_url, api_key=client_api_key, default_headers=default_headers)

async def process_stream(stream):
    first_token_time = None
    total_tokens = 0
    try:
        async for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time()
            
            # 检查是否有内容
            content = chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta, 'content') else None
            reasoning_content = getattr(chunk.choices[0].delta, "reasoning_content", None)
            
            if content or reasoning_content:
                total_tokens += 1
            
            # 检查是否完成
            if chunk.choices[0].finish_reason is not None:
                logging.debug(f"流式响应完成，原因: {chunk.choices[0].finish_reason}")
                break
        
        logging.debug(f"流式响应处理完毕，共收到 {total_tokens} 个token")
        return first_token_time, total_tokens
    except Exception as e:
        logging.error(f"处理流式响应时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # 如果已经收到了一些token，返回已有数据；否则返回None
        if first_token_time and total_tokens > 0:
            return first_token_time, total_tokens
        else:
            raise  # 重新抛出异常，让上层函数处理

async def make_request(client, model, output_tokens, request_timeout, use_long_context):
    start_time = time.time()
    if use_long_context:
        prompt_pair = random.choice(LONG_PROMPT_PAIRS)
        content = prompt_pair["context"] + "\n\n" + prompt_pair["prompt"]
    else:
        content = random.choice(SHORT_PROMPTS)

    # 记录请求参数 - 保留这条有用的日志，但简化内容
    logging.debug(f"请求参数: model={model}, max_tokens={output_tokens}, use_long_context={use_long_context}")
    
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=output_tokens,
            stream=True
        )
        
        first_token_time, total_tokens = await asyncio.wait_for(process_stream(stream), timeout=request_timeout)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        ttft = first_token_time - start_time if first_token_time else None
        tokens_per_second = total_tokens / elapsed_time if elapsed_time > 0 else 0
        
        # 使用更有意义的日志信息
        logging.info(f"请求成功: tokens={total_tokens}, 耗时={elapsed_time:.2f}秒, TPS={tokens_per_second:.2f}, TTFT={ttft:.3f}秒")
        return total_tokens, elapsed_time, tokens_per_second, ttft, "success", None

    except asyncio.TimeoutError:
        logging.warning(f"请求超时: 超过{request_timeout}秒")
        return None, None, None, None, "timeout", f"请求超时（{request_timeout}秒）"
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # 更详细的错误分类
        if "rate_limit" in error_msg.lower():
            error_category = "rate_limit"
        elif "auth" in error_msg.lower() or "key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            error_category = "auth_error"
        elif "connect" in error_msg.lower() or "network" in error_msg.lower() or "connection" in error_msg.lower():
            error_category = "network_error"
        elif "not found" in error_msg.lower() or "404" in error_msg:
            error_category = "not_found"
        elif "invalid" in error_msg.lower() or "parameter" in error_msg.lower():
            error_category = "invalid_params"
        else:
            error_category = "api_error"
        
        # 简化错误日志，但保留关键信息
        logging.error(f"请求失败({error_category}): {error_type}: {error_msg}")
        
        return None, None, None, None, error_category, f"{error_type}: {error_msg}"

async def worker(client, semaphore, queue, results, model, output_tokens, request_timeout, use_long_context):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break
            logging.debug(f"Starting request {task_id}")
            result = await make_request(client, model, output_tokens, request_timeout, use_long_context)
            if result:
                results.append(result)
                if result[4] != "success":  # 如果不是成功状态
                    logging.warning(f"Request {task_id} failed with error type: {result[4]}, message: {result[5]}")
            else:
                logging.warning(f"Request {task_id} failed with unknown error")
            queue.task_done()
            logging.debug(f"Finished request {task_id}")

def calculate_percentile(values, percentile, reverse=False):
    if not values:
        return None
    if reverse:
        return np.percentile(values, 100 - percentile)
    return np.percentile(values, percentile)

async def run_benchmark(num_requests, concurrency, request_timeout, output_tokens, llm_url, api_key, model, use_long_context, auth_config=None):
    client = _create_llm_client(llm_url, api_key, auth_config)
    semaphore = asyncio.Semaphore(concurrency)
    queue = asyncio.Queue()
    results = []

    # Add tasks to the queue
    for i in range(num_requests):
        await queue.put(i)
    
    # Add sentinel values to stop workers
    for _ in range(concurrency):
        await queue.put(None)

    # Create worker tasks
    workers = [asyncio.create_task(worker(client, semaphore, queue, results, model, output_tokens, request_timeout, use_long_context)) for _ in range(concurrency)]

    start_time = time.time()
    
    # Wait for all tasks to complete
    await queue.join()
    await asyncio.gather(*workers)

    end_time = time.time()

    # Calculate metrics
    total_elapsed_time = end_time - start_time
    total_tokens = sum(tokens for tokens, _, _, _, _, _ in results if tokens is not None)
    latencies = [elapsed_time for _, elapsed_time, _, _, _, _ in results if elapsed_time is not None]
    tokens_per_second_list = [tps for _, _, tps, _, _, _ in results if tps is not None]
    ttft_list = [ttft for _, _, _, ttft, _, _ in results if ttft is not None]

    # 收集错误统计
    error_counter = collections.Counter()
    error_samples = {}
    for _, _, _, _, status, error_msg in results:
        if status != "success":
            error_counter[status] += 1
            # 为每种错误类型保存最多3个样本
            if status not in error_samples:
                error_samples[status] = []
            if len(error_samples[status]) < 3 and error_msg:
                error_samples[status].append(error_msg)

    successful_requests = sum(1 for _, _, _, _, status, _ in results if status == "success")
    failed_requests = num_requests - successful_requests
    requests_per_second = successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    
    # Calculate percentiles
    percentiles = [50, 95, 99]
    latency_percentiles = [calculate_percentile(latencies, p) for p in percentiles]
    tps_percentiles = [calculate_percentile(tokens_per_second_list, p, reverse=True) for p in percentiles]
    ttft_percentiles = [calculate_percentile(ttft_list, p) for p in percentiles]
    
    return {
        "total_requests": num_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "concurrency": concurrency,
        "request_timeout": request_timeout,
        "max_output_tokens": output_tokens,
        "use_long_context": use_long_context,
        "model": model,
        "total_time": total_elapsed_time,
        "requests_per_second": requests_per_second,
        "total_output_tokens": total_tokens,
        "error_statistics": {
            "count": dict(error_counter),
            "samples": error_samples
        },
        "latency": {
            "average": avg_latency,
            "p50": latency_percentiles[0],
            "p95": latency_percentiles[1],
            "p99": latency_percentiles[2]
        },
        "tokens_per_second": {
            "average": avg_tokens_per_second,
            "p50": tps_percentiles[0],
            "p95": tps_percentiles[1],
            "p99": tps_percentiles[2]
        },
        "time_to_first_token": {
            "average": avg_ttft,
            "p50": ttft_percentiles[0],
            "p95": ttft_percentiles[1],
            "p99": ttft_percentiles[2]
        }
    }

def print_results(results, output_format="both"):
    """
    打印测试结果
    :param results: 测试结果字典
    :param output_format: 输出格式 ('json', 'line', 'both')
    """
    if output_format in ['json', 'both']:
        print(json.dumps(results, indent=2))
    
    if output_format in ['line', 'both']:
        print("\n基本信息:")
        print(f"总请求数: {results.get('total_requests', 0)} 个")
        print(f"成功请求数: {results.get('successful_requests', 0)} 个")
        print(f"并发数: {results.get('concurrency', 0)} 个")
        print(f"请求超时: {results.get('request_timeout', 0)} 秒")
        print(f"最大输出token数: {results.get('max_output_tokens', 0)}")
        print(f"是否使用长文本: {'是' if results.get('use_long_context', False) else '否'}")
        
        # 安全获取数值，提供默认值
        total_time = results.get('total_time', 0)
        rps = results.get('requests_per_second', 0)
        total_tokens = results.get('total_output_tokens', 0)
        model = results.get('model', '未知')
        
        print(f"总运行时间: {total_time:.2f} 秒")
        print(f"每秒请求数 (RPS): {rps:.2f}")
        print(f"总输出token数: {total_tokens}")
        print(f"模型名称: {model}")
        
        # 安全获取延迟数据
        latency_data = results.get('latency', {})
        if not isinstance(latency_data, dict):
            latency_data = {}
            
        avg_latency = latency_data.get('average', 0)
        p50_latency = latency_data.get('p50', 0)
        p95_latency = latency_data.get('p95', 0)
        p99_latency = latency_data.get('p99', 0)
        
        print("\n延迟统计 (单位: 秒):")
        print(f"平均延迟: {avg_latency:.3f}")
        print(f"延迟 P50: {p50_latency:.3f}" if p50_latency is not None else "延迟 P50: N/A")
        print(f"延迟 P95: {p95_latency:.3f}" if p95_latency is not None else "延迟 P95: N/A")
        print(f"延迟 P99: {p99_latency:.3f}" if p99_latency is not None else "延迟 P99: N/A")
        
        # 安全获取TPS数据
        tps_data = results.get('tokens_per_second', {})
        if not isinstance(tps_data, dict):
            tps_data = {}
            
        avg_tps = tps_data.get('average', 0)
        p50_tps = tps_data.get('p50', 0)
        p95_tps = tps_data.get('p95', 0)
        p99_tps = tps_data.get('p99', 0)
        
        print("\nToken生成速度 (tokens/sec):")
        print(f"平均TPS: {avg_tps:.2f}")
        print(f"TPS P50: {p50_tps:.2f}" if p50_tps is not None else "TPS P50: N/A")
        print(f"TPS P95: {p95_tps:.2f}" if p95_tps is not None else "TPS P95: N/A")
        print(f"TPS P99: {p99_tps:.2f}" if p99_tps is not None else "TPS P99: N/A")
        
        # 安全获取TTFT数据
        ttft_data = results.get('time_to_first_token', {})
        if not isinstance(ttft_data, dict):
            ttft_data = {}
            
        avg_ttft = ttft_data.get('average', 0)
        p50_ttft = ttft_data.get('p50', 0)
        p95_ttft = ttft_data.get('p95', 0)
        p99_ttft = ttft_data.get('p99', 0)
        
        print("\n首Token延迟 (秒):")
        print(f"平均TTFT: {avg_ttft:.3f}")
        print(f"TTFT P50: {p50_ttft:.3f}" if p50_ttft is not None else "TTFT P50: N/A")
        print(f"TTFT P95: {p95_ttft:.3f}" if p95_ttft is not None else "TTFT P95: N/A")
        print(f"TTFT P99: {p99_ttft:.3f}" if p99_ttft is not None else "TTFT P99: N/A")
        
        # 错误统计
        if 'error_statistics' in results and results['error_statistics'].get('count'):
            print("\n错误统计:")
            for error_type, count in results['error_statistics']['count'].items():
                print(f"{error_type}: {count}")
            
            # 错误样本
            if results['error_statistics'].get('samples'):
                print("\n错误样本:")
                for error_type, samples in results['error_statistics']['samples'].items():
                    print(f"{error_type} 样本:")
                    for sample in samples:
                        print(sample)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ai model with LLM")
    parser.add_argument("--num_requests", type=int, required=True, help="Number of requests to make")
    parser.add_argument("--concurrency", type=int, required=True, help="Number of concurrent requests")
    parser.add_argument("--request_timeout", type=int, default=60, help="Timeout for each request in seconds (default: 60)")
    parser.add_argument("--output_tokens", type=int, default=50, help="Number of output tokens (default: 50)")
    parser.add_argument("--llm_url", type=str, required=True, help="URL of the LLM server")
    parser.add_argument("--api_key", type=str, required=False, default="default", help="API key for LLM server")
    parser.add_argument("--model", type=str, default="deepseek-r1", 
                       help="Model name to use for inference (default: deepseek-r1)")
    parser.add_argument("--use_long_context", action="store_true", help="Use long context prompt pairs instead of short prompts")
    parser.add_argument("--auth_type", type=str, choices=['auto', 'bearer', 'basic', 'none'], default='auto',
                       help="Authentication strategy. auto=detect based on provided credentials.")
    parser.add_argument("--basic_auth_user", type=str, help="Username for HTTP Basic auth")
    parser.add_argument("--basic_auth_password", type=str, help="Password for HTTP Basic auth")
    parser.add_argument("--auth_header", type=str, help="Override Authorization header (e.g. 'Basic xxxx')")
    parser.add_argument("--output_format", type=str, choices=['json', 'line', 'both'], 
                       default='line', help="Output format (json/line/both)")
    args = parser.parse_args()

    auth_config = {
        "auth_type": args.auth_type,
        "basic_auth_user": args.basic_auth_user,
        "basic_auth_password": args.basic_auth_password,
        "auth_header": args.auth_header,
    }

    results = asyncio.run(run_benchmark(
        args.num_requests, 
        args.concurrency, 
        args.request_timeout, 
        args.output_tokens, 
        args.llm_url, 
        args.api_key,
        args.model,
        args.use_long_context,
        auth_config
    ))
    print_results(results, args.output_format)

else:
    # When imported as a module, provide the run_benchmark function
    __all__ = ['run_benchmark']
