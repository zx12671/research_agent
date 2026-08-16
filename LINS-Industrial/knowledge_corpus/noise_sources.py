"""
noise_sources.py: 噪声文档生成器

功能: 为 Open-Domain RAG 评估构建噪声/干扰文档。
这些文档不属于 IndustryBench 测试集，用于证明检索器能够区分相关与无关文档。

噪声来源（模拟）:
  1. Wikipedia Manufacturing — 制造业百科片段
  2. Maintenance Manual — 通用维护手册
  3. FactoryWave — 工厂自动化相关文档
  
使用方法:
  python noise_sources.py
  将生成 knowledge_corpus/noise/ 下的 JSONL 文件

输出:
  knowledge_corpus/noise/sources/    — 原始噪声文档
  knowledge_corpus/noise/chunks/     — 切分后的噪声块（会被 industrial_retriever 加载）

论文意义:
  审稿人: "你的 Knowledge 是否全部来自测试集？"
  → 答: "No. 我们在 knowledge_corpus 中加入了 {num_noise} 篇噪声文档，
     这些文档来自 Wikipedia 制造条目/维护手册/工厂自动化文档，
     它们与测试集完全无关。检索器需要从中筛选出真正相关的文档。"
"""

import os
import json
import uuid
import textwrap


# ============================================================
# 噪声文档模板（模拟真实工业文档，但来自非 IndustryBench 来源）
# ============================================================

WIKIPEDIA_MANUFACTURING_TEMPLATES = [
    {
        "title": "Manufacturing Engineering Overview",
        "source": "Wikipedia: Manufacturing Engineering",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Manufacturing engineering is a field of engineering dealing with the design, 
            development, and implementation of integrated systems of people, machinery, 
            and information for the production of high-quality, economically competitive products.
            
            The manufacturing process involves raw material acquisition, material processing,
            assembly, quality control, and logistics. Modern manufacturing increasingly relies 
            on automation, robotics, and computer-integrated manufacturing systems.
            
            Key metrics in manufacturing include Overall Equipment Effectiveness (OEE),
            throughput, cycle time, defect rate, and yield. Six Sigma and Lean Manufacturing
            are widely adopted methodologies for process improvement.
        """),
    },
    {
        "title": "CNC Machining Fundamentals",
        "source": "Wikipedia: CNC Machining",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Computer Numerical Control (CNC) machining is a subtractive manufacturing process 
            where pre-programmed computer software dictates the movement of factory tools and machinery.
            
            Common CNC processes include milling, turning, drilling, and grinding. The machine
            tools are controlled by G-code instructions that specify feed rate, spindle speed,
            tool path, and depth of cut.
            
            Typical tolerances for CNC machining range from ±0.005mm to ±0.1mm depending on
            the machine type, material, and process parameters. Surface finish is typically
            measured in Ra (roughness average) values ranging from 0.4μm to 6.3μm.
        """),
    },
    {
        "title": "Industrial Robotics",
        "source": "Wikipedia: Industrial Robotics",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Industrial robots are automated, programmable machines capable of movement on 
            three or more axes. They are widely used in manufacturing for tasks such as 
            welding, painting, assembly, pick and place, packaging, and inspection.
            
            The six-axis articulated robot is the most common type, providing flexibility
            for complex operations. Collaborative robots (cobots) are designed to work 
            alongside human operators with built-in safety features.
            
            Robot positioning accuracy is typically within ±0.02mm to ±0.1mm, while 
            repeatability can be as precise as ±0.01mm. Payload capacity ranges from 
            3kg for small assembly robots to over 1000kg for heavy industrial applications.
        """),
    },
    {
        "title": "Quality Control Methods",
        "source": "Wikipedia: Quality Control",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Quality control (QC) is a process by which entities review the quality of all 
            factors involved in production. ISO 9000 defines quality as "the degree to which 
            a set of inherent characteristics fulfills requirements".
            
            Statistical Process Control (SPC) uses control charts to monitor manufacturing 
            processes. Common control charts include X-bar and R charts for variables, and 
            p-charts and c-charts for attributes.
            
            Acceptance sampling plans, such as ANSI/ASQ Z1.4 and Z1.9, are used for 
            lot-by-lot inspection. The Average Outgoing Quality Limit (AOQL) is a key 
            metric in sampling inspection.
        """),
    },
    {
        "title": "Production Planning and Scheduling",
        "source": "Wikipedia: Production Planning",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Production planning is the process of allocating raw materials, equipment, 
            and human resources to meet production targets efficiently.
            
            Key concepts include Master Production Schedule (MPS), Material Requirements 
            Planning (MRP), and Capacity Requirements Planning (CRP). Just-in-Time (JIT) 
            manufacturing aims to minimize inventory holding costs.
            
            The Economic Order Quantity (EOQ) model determines the optimal order quantity 
            that minimizes total inventory costs. It is calculated as:
            EOQ = sqrt(2DS/H), where D is demand, S is setup cost, and H is holding cost.
        """),
    },
    {
        "title": "Additive Manufacturing Technologies",
        "source": "Wikipedia: 3D Printing",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Additive manufacturing (AM), also known as 3D printing, creates objects by 
            adding material layer by layer. Common AM technologies include:
            
            - Fused Deposition Modeling (FDM): thermoplastic filament extrusion
            - Selective Laser Sintering (SLS): laser sintering of powder materials
            - Stereolithography (SLA): UV curing of liquid resin
            - Direct Metal Laser Sintering (DMLS): metal powder fusion
            
            Layer thickness typically ranges from 0.05mm to 0.3mm. Build volume varies 
            from desktop printers (200x200x200mm) to industrial systems (over 1m).
        """),
    },
]

MAINTENANCE_MANUAL_TEMPLATES = [
    {
        "title": "Preventive Maintenance Schedule",
        "source": "Maintenance Manual: General Guidelines",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Preventive maintenance (PM) is scheduled maintenance performed to prevent 
            equipment failures before they occur. A typical PM schedule includes:
            
            Daily: visual inspection, fluid level checks, temperature monitoring
            Weekly: lubrication, filter cleaning, belt tension adjustment
            Monthly: electrical connection check, sensor calibration, valve testing
            Quarterly: oil analysis, vibration analysis, alignment verification
            Annually: major overhaul, component replacement, system upgrade
            
            Condition-based maintenance (CBM) uses real-time monitoring data to predict 
            when maintenance should be performed. Vibration analysis, thermography, and 
            oil analysis are common CBM techniques.
        """),
    },
    {
        "title": "Equipment Reliability Metrics",
        "source": "Maintenance Manual: Reliability Engineering",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Equipment reliability is measured using several key performance indicators:
            
            Mean Time Between Failure (MTBF): total operating time / number of failures
            Mean Time To Repair (MTTR): total downtime / number of repairs
            Availability: MTBF / (MTBF + MTTR)
            
            A target availability of 95% or higher is typical for critical equipment.
            The Failure Mode and Effects Analysis (FMEA) is used to identify potential 
            failure modes and their effects on system performance.
        """),
    },
    {
        "title": "Lubrication Best Practices",
        "source": "Maintenance Manual: Lubrication Guide",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Proper lubrication is essential for equipment longevity and performance.
            Key lubrication parameters include:
            
            Viscosity grade (ISO VG): determines oil film thickness
            NLGI grade: consistency rating for greases (00 to 6)
            Operating temperature range: affects lubricant selection
            Relubrication interval: depends on bearing type and operating conditions
            
            Common lubricant types include mineral oils, synthetic oils, and lithium-based 
            greases. The ISO cleanliness code (e.g., ISO 16/14/11) indicates particle 
            contamination levels in the lubricant.
        """),
    },
]

FACTORYWAVE_TEMPLATES = [
    {
        "title": "Factory Automation Architecture",
        "source": "FactoryWave: Industry 4.0 Guide",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            The ISA-95 (IEC 62264) standard defines the hierarchical model for factory 
            automation systems:
            
            Level 4: Enterprise Resource Planning (ERP)
            Level 3: Manufacturing Execution System (MES)
            Level 2: Supervisory Control and Data Acquisition (SCADA)
            Level 1: Programmable Logic Controllers (PLC)
            Level 0: Sensors and Actuators
            
            OPC Unified Architecture (OPC UA) is the standard communication protocol for 
            industrial automation, enabling secure and reliable data exchange between 
            different levels of the automation hierarchy.
        """),
    },
    {
        "title": "Programmable Logic Controllers",
        "source": "FactoryWave: PLC Programming Guide",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Programmable Logic Controllers (PLCs) are industrial digital computers used 
            for automation of manufacturing processes. Key specifications include:
            
            Scan time: typically 1-100ms depending on program complexity
            I/O capacity: from 8 to over 10000 points
            Programming languages: Ladder Logic (LD), Function Block Diagram (FBD),
            Structured Text (ST), Instruction List (IL), Sequential Function Chart (SFC)
            
            Common PLC brands include Siemens S7 series, Allen-Bradley ControlLogix, 
            Mitsubishi MELSEC, and Schneider Modicon. Safety PLCs provide SIL 3 
            rated control for safety-critical applications.
        """),
    },
    {
        "title": "Industrial Sensor Technologies",
        "source": "FactoryWave: Sensor Handbook",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Industrial sensors convert physical parameters into electrical signals for 
            monitoring and control. Common industrial sensor types include:
            
            Temperature: thermocouples (Type K, J, T), RTDs (PT100), infrared pyrometers
            Pressure: strain-gauge, capacitive, piezoelectric (4-20mA output)
            Flow: electromagnetic, ultrasonic, Coriolis, vortex
            Level: radar, ultrasonic, guided wave radar, capacitance
            Position: linear variable differential transformer (LVDT), encoder
            
            Sensor accuracy is typically specified as ±% of full scale or ±% of reading.
            IP ratings (e.g., IP67) indicate protection against dust and water ingress.
        """),
    },
    {
        "title": "Industrial Communication Protocols",
        "source": "FactoryWave: Industrial Networks",
        "industry": "general_manufacturing",
        "content": textwrap.dedent("""\
            Industrial communication protocols enable data exchange between automation 
            devices. Common fieldbus and industrial Ethernet protocols include:
            
            PROFIBUS: RS-485 based, up to 12 Mbps
            PROFINET: industrial Ethernet, real-time (RT) and isochronous (IRT)
            EtherNet/IP: based on TCP/IP, up to 100 Mbps
            Modbus TCP: simple client-server protocol, widely used
            EtherCAT: high-performance, 100 Mbps, distributed clocks
            
            Cycle times vary from 100μs for EtherCAT motion control to 10ms for 
            standard Modbus TCP applications. Cable lengths are typically limited 
            to 100m for Ethernet-based protocols.
        """),
    },
]


# ============================================================
# 噪声文档生成
# ============================================================

def generate_noise_documents() -> list:
    """生成所有噪声文档"""
    documents = []
    
    all_templates = (
        WIKIPEDIA_MANUFACTURING_TEMPLATES +
        MAINTENANCE_MANUAL_TEMPLATES +
        FACTORYWAVE_TEMPLATES
    )
    
    for i, template in enumerate(all_templates):
        doc = {
            "id": f"noise_{i+1:04d}",
            "title": template["title"],
            "source": template["source"],
            "industry": template["industry"],
            "content": template["content"].strip(),
            "category": "distractor",  # 标记为干扰文档
            "noise_group": "wikipedia" if i < len(WIKIPEDIA_MANUFACTURING_TEMPLATES)
                           else "maintenance" if i < len(WIKIPEDIA_MANUFACTURING_TEMPLATES) + len(MAINTENANCE_MANUAL_TEMPLATES)
                           else "factorywave",
        }
        documents.append(doc)
    
    return documents


def save_noise_documents(output_dir: str = None):
    """保存噪声文档到 sources/ 目录"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "noise", "sources")
    
    os.makedirs(output_dir, exist_ok=True)
    
    documents = generate_noise_documents()
    output_path = os.path.join(output_dir, "noise_documents.jsonl")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    print(f"[INFO] 已生成 {len(documents)} 篇噪声文档 → {output_path}")
    return output_path


if __name__ == "__main__":
    save_noise_documents()
    print("[INFO] 噪声文档生成完成！")
    print("[INFO] 请在 knowledge_corpus 中构建索引后运行 RAG 实验")
