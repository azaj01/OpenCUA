
<h1 style="
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  font-size:48px;
  font-weight:700;
  line-height:1.25;
  text-align:center;
  margin:0 0 24px;">
  OpenCUA: Open Foundations for <br> Computer-Use Agents
</h1>

<div style="
  display:flex;
  justify-content:center;
  gap:12px;
  flex-wrap:wrap;
  margin-bottom:28px;">
  
  <a href="https://opencua.xlang.ai/" style="
     display:inline-block;
     padding:8px 24px;
     background:#2b2b2b;
     color:#ffffff;
     border-radius:36px;
     text-decoration:none;
     font-weight:600;
     font-size:16px;">
    🌐 Website
  </a>

  <a href="https://github.com/XinyuanWangCS/OpenCUA-Website/blob/main/static/pdf/OpenCUA_arxiv.pdf" style="
     display:inline-block;
     padding:8px 24px;
     background:#2b2b2b;
     color:#ffffff;
     border-radius:36px;
     text-decoration:none;
     font-weight:600;
     font-size:16px;">
    📝 Paper
  </a>

  <a href="https://github.com/xlang-ai/OpenCUA" style="
     display:inline-block;
     padding:8px 24px;
     background:#2b2b2b;
     color:#ffffff;
     border-radius:36px;
     text-decoration:none;
     font-weight:600;
     font-size:16px;">
    💻 Code
  </a>
</div>

<div style="max-width:900px;margin:0 auto;">


# 1. Introduction
<div style="
  max-width: 880px;              /* 可按需调节整体宽度 */
  margin: 0 auto;               /* 居中容器 */
  text-align: justify;          /* 关键：两端对齐 */
  text-justify: inter-word;     /* 优化英文对齐效果 */
  line-height: 1.6;">
  
Vision-language models have demonstrated impressive capabilities as computer-use agents (CUAs) capable of automating diverse computer tasks. 
As their commercial potential grows, critical details of the most capable CUA systems remain closed. As these agents will increasingly mediate digital interactions and execute consequential decisions on our behalf, 
the research community needs access to open CUA frameworks to study their capabilities, limitations, and risks. 
To bridge this gap, we propose <b>OpenCUA</b>, a comprehensive open-source framework for scaling CUA data and foundation models. 
Our framework consists of: (1) an annotation infrastructure that seamlessly captures human computer-use demonstrations; 
(2) <b>AgentNet</b>, the first large-scale computer-use task dataset spanning 3 operating systems and 200+ applications and websites; 
(3) a scalable pipeline that transforms demonstrations into state–action pairs with reflective long Chain-of-Thought reasoning that sustain robust performance gains as data scales. 
Our end-to-end agent models demonstrate strong performance across CUA benchmarks. In particular, <b>OpenCUA-32B</b> achieves an average success rate of 32.5% on **[OSWorld-Verified](https://os-world.github.io/)**, 
establishing a new state-of-the-art (SOTA) among open-source models and surpassing OpenAI CUA (GPT-4o). 
Further analysis confirms that our approach generalizes well across domains and benefits significantly from increased test-time computation. 
We release our annotation tool, datasets, code, and models to build open foundations for further CUA research.
</div>

## 2. AgentNetTool – Annotation & Verification Tool
Our **AgentNetTool** is a cross-platform GUI recorder that runs unobtrusively on annotators’ machines.  
It captures synchronized **screen video**, **mouse/keyboard events**, and **accessibility trees**, then provides an in-browser UI for reviewing, trimming, and submitting demonstrations.  

👉 **AgentNetTool Document:** <https://agentnet-tool.xlang.ai/>

---

## 3. DataProcessor – Action Reduction & State–Action Matching
Raw demonstrations can contain thousands of low-level events that are too dense for model training.  
The **DataProcessor** module (`./DataProcessor/`) performs two key steps:

1. **Action Reduction** — merges granular signals into concise, semantically meaningful PyAutoGUI actions (e.g., collapsing mouse moves → click, coalescing scrolls, grouping key-press sequences into text or hotkeys).  
2. **State–Action Matching** — aligns every reduced action with the *last visually distinct frame* **before** the action begins, avoiding future-information leakage and yielding compact state–action pairs `⟨sᵢ, aᵢ⟩`.

These processed trajectories underlie all downstream training and evaluation.

---

## 4. CoTGenerator – Synthesizing Reflective Long Chain-of-Thought Reasoning
To boost robustness and interpretability, we augment each trajectory with **reflective long Chain-of-Thought (CoT) reasoning**.  
The **CoTGenerator** pipeline (`./CoTGenerator/`) synthesizes step-level reflections that:

* reflect on the previous action,
* explain *why* an action is chosen given the current observation and history,  
* note potential alternative actions, and  
* forecast the expected next state.

Empirically, models trained with these rich CoTs scale better with data and generalize across unseen applications.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Research Use and Disclaimer

This software is intended for **research and educational purposes only**. 

## Citation

If you use OpenCUA in your research, please cite our work:

```bibtex
@article{OpenCUA2025, 
  title={OpenCUA: Open Foundations for Computer-Use Agents}, 
  author={Wang, Xinyuan and Wang, Bowen and Lu, Dunjie and Yang, Junlin and Xie, Tianbao and Wang, Junli and Deng, Jiaqi and Guo, Xiaole and Xu, Yiheng and Wu, Chen Henry and Shen, Zhennan and Li, Zhuokai and Li, Ryan and Li, Xiaochuan and Chen, Junda and Zheng, Boyuan and Li, Peihang and Lei, Fangyu and Cao, Ruisheng and Fu, Yeqiao and Shin, Dongchan and Shin, Martin and Hu, Jiarui and Wang, Yuyan and Chen, Jixuan and Ye, Yuxiao and Zhang, Danyang and Wang, Yipu and Wang, Heng and Yang, Diyi and Zhong, Victor and Charles, Y. and Yang, Zhilin and Yu, Tao}, 
  year={2025}, 
  url={https://opencua.xlang.ai/} 
}
```

</div>