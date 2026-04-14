# verl PPO 初始化调用图：`main_ppo` → `RayPPOTrainer.init_workers()`

## 1. 顶层调用流程

从 `python3 -m verl.trainer.main_ppo` 启动，到 `RayPPOTrainer` 初始化完毕 (含 `init_workers`) 的完整调用链。

> [!IMPORTANT]
> **关键路径** (Critical Path) 用 🔴 红色粗线标注，这些是初始化过程中不可并行、耗时最长的瓶颈步骤。

```mermaid
flowchart TD
    classDef critical fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:3px
    classDef important fill:#ffa94d,stroke:#e8590c,color:#fff,stroke-width:2px
    classDef normal fill:#4dabf7,stroke:#1971c2,color:#fff
    classDef data fill:#69db7c,stroke:#2b8a3e,color:#fff
    classDef config fill:#da77f2,stroke:#9c36b5,color:#fff

    START["__main__<br/>main_ppo.py:444"]:::normal
    HYDRA["@hydra.main()<br/>config_path='config'<br/>config_name='ppo_trainer'"]:::config
    MAIN["main(config)"]:::normal
    AUTO_DEV["auto_set_device(config)<br/>自动检测 NPU/CUDA"]:::config
    MIGRATE["migrate_legacy_reward_impl(config)<br/>旧版 reward 配置迁移"]:::config
    RUN_PPO["run_ppo(config)"]:::important

    START --> HYDRA
    HYDRA -->|"OmegaConf DictConfig"| MAIN
    MAIN --> AUTO_DEV
    AUTO_DEV --> MIGRATE
    MIGRATE -->|"config (已迁移)"| RUN_PPO
```

## 2. `run_ppo` → Ray 集群初始化 & TaskRunner 远程执行

```mermaid
flowchart TD
    classDef critical fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:3px
    classDef important fill:#ffa94d,stroke:#e8590c,color:#fff,stroke-width:2px
    classDef normal fill:#4dabf7,stroke:#1971c2,color:#fff
    classDef data fill:#69db7c,stroke:#2b8a3e,color:#fff
    classDef ray fill:#be4bdb,stroke:#862e9c,color:#fff

    RUN_PPO["run_ppo(config)"]:::important

    subgraph RAY_INIT ["Ray Cluster 初始化"]
        direction TB
        CHECK_RAY{"ray.is_initialized()?"}:::normal
        GET_ENV["get_ppo_ray_runtime_env()<br/>设置 TOKENIZERS_PARALLELISM,<br/>NCCL_DEBUG, VLLM 等环境变量"]:::normal
        MERGE_ENV["OmegaConf.merge(default, user)"]:::normal
        RAY_START["ray.init(**ray_init_kwargs)"]:::critical
    end

    subgraph REMOTE_EXEC ["Remote Task Execution"]
        direction TB
        WRAP_CLS["ray.remote(num_cpus=1)(TaskRunner)"]:::ray
        CREATE_RUNNER["runner = task_runner_class.remote()"]:::ray
        RAY_GET["ray.get(runner.run.remote(config))"]:::critical
    end

    RUN_PPO --> CHECK_RAY
    CHECK_RAY -->|No| GET_ENV
    GET_ENV --> MERGE_ENV
    MERGE_ENV --> RAY_START
    RAY_START --> WRAP_CLS
    CHECK_RAY -->|Yes| WRAP_CLS
    WRAP_CLS --> CREATE_RUNNER
    CREATE_RUNNER -->|"config (序列化)"| RAY_GET
```

## 3. `TaskRunner.run()` — 完整的初始化编排

这是在 **Ray Worker 节点**上远程执行的主初始化逻辑。

```mermaid
flowchart TD
    classDef critical fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:3px
    classDef important fill:#ffa94d,stroke:#e8590c,color:#fff,stroke-width:2px
    classDef normal fill:#4dabf7,stroke:#1971c2,color:#fff
    classDef data fill:#69db7c,stroke:#2b8a3e,color:#fff
    classDef config fill:#da77f2,stroke:#9c36b5,color:#fff

    TR_RUN["TaskRunner.run(config)"]:::critical

    RESOLVE["OmegaConf.resolve(config)"]:::config
    ADD_AR["add_actor_rollout_worker(config)<br/>🔴 关键路径"]:::critical
    ADD_CR["add_critic_worker(config)"]:::important
    ADD_RM["add_reward_model_resource_pool(config)"]:::normal
    ADD_REF["add_ref_policy_worker(config, cls)"]:::normal
    VALIDATE["validate_config(config)"]:::config
    COPY_LOCAL["copy_to_local(model_path)<br/>下载模型到本地"]:::critical
    TOKENIZER["hf_tokenizer(local_path)"]:::data
    PROCESSOR["hf_processor(local_path)<br/>多模态用, 可为 None"]:::data
    INIT_RPM["init_resource_pool_mgr(config)"]:::important
    CREATE_TRAIN["create_rl_dataset(train)"]:::data
    CREATE_VAL["create_rl_dataset(val)"]:::data
    CREATE_SAMPLER["create_rl_sampler(config)"]:::data
    TRAINER_INIT["RayPPOTrainer(...)"]:::critical
    INIT_WORKERS["trainer.init_workers()<br/>🔴 关键路径"]:::critical
    FIT["trainer.fit()"]:::normal

    TR_RUN --> RESOLVE
    RESOLVE --> ADD_AR
    ADD_AR -->|"actor_rollout_cls,<br/>ray_worker_group_cls"| ADD_CR
    ADD_CR --> ADD_RM
    ADD_RM --> ADD_REF
    ADD_REF --> VALIDATE
    VALIDATE --> COPY_LOCAL
    COPY_LOCAL -->|"local_path"| TOKENIZER
    TOKENIZER --> PROCESSOR
    PROCESSOR --> INIT_RPM
    INIT_RPM -->|"resource_pool_manager"| CREATE_TRAIN
    CREATE_TRAIN --> CREATE_VAL
    CREATE_VAL --> CREATE_SAMPLER
    CREATE_SAMPLER --> TRAINER_INIT
    TRAINER_INIT --> INIT_WORKERS
    INIT_WORKERS --> FIT

    style FIT stroke-dasharray:5 5
```

## 4. Worker 角色映射构建

`TaskRunner` 的几个 `add_*` 方法负责填充 `role_worker_mapping` 和 `mapping` 两个字典。

```mermaid
flowchart LR
    classDef critical fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:3px
    classDef important fill:#ffa94d,stroke:#e8590c,color:#fff,stroke-width:2px
    classDef normal fill:#4dabf7,stroke:#1971c2,color:#fff
    classDef worker fill:#51cf66,stroke:#2b8a3e,color:#fff
    classDef role fill:#fcc419,stroke:#e67700,color:#333

    subgraph ROLE_MAP ["role_worker_mapping"]
        direction TB
        R_AR["Role.ActorRollout<br/>或 Role.ActorRolloutRef"]:::role
        R_CR["Role.Critic"]:::role
        R_REF["Role.RefPolicy"]:::role
        R_RM["Role.RewardModel"]:::role
    end

    subgraph WORKER_CLS ["Worker Classes"]
        direction TB
        W_AR_NEW["ActorRolloutRefWorker<br/>(engine_workers)"]:::worker
        W_AR_FSDP["AsyncActorRolloutRefWorker<br/>(fsdp_workers)"]:::worker
        W_AR_MEGA["AsyncActorRolloutRefWorker<br/>(megatron_workers)"]:::worker
        W_CR_FSDP["CriticWorker / TrainingWorker"]:::worker
        W_CR_MEGA["CriticWorker (megatron)"]:::worker
    end

    subgraph POOL_MAP ["mapping → 资源池ID"]
        direction TB
        P_GLOBAL["'global_pool'"]:::normal
        P_REWARD["'reward_pool'<br/>(可选独立池)"]:::normal
    end

    R_AR -->|"use_legacy='disable'"| W_AR_NEW
    R_AR -->|"strategy='fsdp'"| W_AR_FSDP
    R_AR -->|"strategy='megatron'"| W_AR_MEGA
    R_CR -->|"fsdp+legacy"| W_CR_FSDP
    R_CR -->|"megatron"| W_CR_MEGA

    R_AR -.-> P_GLOBAL
    R_CR -.-> P_GLOBAL
    R_REF -.-> P_GLOBAL
    R_RM -.->|"enable_resource_pool?"| P_REWARD
    R_RM -.->|"otherwise"| P_GLOBAL
```

### Actor/Rollout Worker 选择逻辑

```mermaid
flowchart TD
    classDef critical fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:3px
    classDef normal fill:#4dabf7,stroke:#1971c2,color:#fff
    classDef worker fill:#51cf66,stroke:#2b8a3e,color:#fff
    classDef decision fill:#fcc419,stroke:#e67700,color:#333

    CHECK_LEGACY{"use_legacy_worker_impl"}:::decision
    NEW_ENGINE["ActorRolloutRefWorker<br/>(engine_workers)<br/>Ref 融合在同一 Worker"]:::worker
    CHECK_LORA{"lora_rank > 0?"}:::decision
    NEED_REF{"need_reference_policy?"}:::decision
    ROLE_ARREF["Role = ActorRolloutRef"]:::normal
    ROLE_AR["Role = ActorRollout"]:::normal

    CHECK_FSDP{"strategy='fsdp'?"}:::decision
    FSDP_WORKER["AsyncActorRolloutRefWorker<br/>(fsdp_workers)"]:::worker
    CHECK_MEGA{"strategy='megatron'?"}:::decision
    MEGA_WORKER["AsyncActorRolloutRefWorker<br/>(megatron_workers)"]:::worker
    ROLE_AR_LEGACY["Role = ActorRollout"]:::normal

    CHECK_LEGACY -->|"disable"| NEW_ENGINE
    NEW_ENGINE --> CHECK_LORA
    CHECK_LORA -->|"No"| NEED_REF
    CHECK_LORA -->|"Yes: ref_in_actor"| ROLE_AR
    NEED_REF -->|"Yes"| ROLE_ARREF
    NEED_REF -->|"No"| ROLE_AR

    CHECK_LEGACY -->|"auto / enable"| CHECK_FSDP
    CHECK_FSDP -->|"Yes"| FSDP_WORKER
    CHECK_FSDP -->|"No"| CHECK_MEGA
    CHECK_MEGA -->|"Yes"| MEGA_WORKER
    FSDP_WORKER --> ROLE_AR_LEGACY
    MEGA_WORKER --> ROLE_AR_LEGACY
```

## 5. `RayPPOTrainer.__init__()` — 构造函数

```mermaid
flowchart TD
    classDef critical fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:3px
    classDef important fill:#ffa94d,stroke:#e8590c,color:#fff,stroke-width:2px
    classDef normal fill:#4dabf7,stroke:#1971c2,color:#fff
    classDef data fill:#69db7c,stroke:#2b8a3e,color:#fff
    classDef config fill:#da77f2,stroke:#9c36b5,color:#fff

    CTOR["RayPPOTrainer.__init__()"]:::critical

    STORE_BASE["存储基础属性<br/>tokenizer, processor, config"]:::normal
    CHECK_HYBRID["assert hybrid_engine == True"]:::config
    STORE_MAP["存储 role_worker_mapping,<br/>resource_pool_manager"]:::normal
    DECIDE_FLAGS["计算关键标志位:<br/>use_reference_policy<br/>use_rm<br/>use_critic<br/>ref_in_actor"]:::important
    KL_CTRL{"use_kl_in_reward?"}:::config
    INIT_KL["kl_ctrl_in_reward =<br/>AdaptiveKLController"]:::normal
    CREATE_DL["_create_dataloader()"]:::data

    CTOR --> STORE_BASE
    STORE_BASE --> CHECK_HYBRID
    CHECK_HYBRID --> STORE_MAP
    STORE_MAP --> DECIDE_FLAGS
    DECIDE_FLAGS --> KL_CTRL
    KL_CTRL -->|"Yes"| INIT_KL
    KL_CTRL -->|"No"| CREATE_DL
    INIT_KL --> CREATE_DL

    subgraph DL ["_create_dataloader()"]
        direction TB
        TRAIN_DL["StatefulDataLoader(train)"]:::data
        VAL_DL["StatefulDataLoader(val)"]:::data
        CALC_STEPS["total_training_steps =<br/>len(train_dl) × total_epochs"]:::normal
        SET_OPTIM["写入 actor/critic optim<br/>total_training_steps"]:::config
    end

    CREATE_DL --> TRAIN_DL
    TRAIN_DL --> VAL_DL
    VAL_DL --> CALC_STEPS
    CALC_STEPS --> SET_OPTIM
```

## 6. `RayPPOTrainer.init_workers()` — 🔴 核心初始化路径

这是耗时最长、逻辑最复杂的初始化步骤。

```mermaid
flowchart TD
    classDef critical fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:3px
    classDef important fill:#ffa94d,stroke:#e8590c,color:#fff,stroke-width:2px
    classDef normal fill:#4dabf7,stroke:#1971c2,color:#fff
    classDef data fill:#69db7c,stroke:#2b8a3e,color:#fff
    classDef ray fill:#be4bdb,stroke:#862e9c,color:#fff

    IW["init_workers()"]:::critical

    subgraph PHASE1 ["Phase 1: 创建资源池"]
        direction TB
        CREATE_RP["resource_pool_manager<br/>.create_resource_pool()<br/>创建 PlacementGroup"]:::critical
        INIT_CLS_MAP["初始化<br/>resource_pool_to_cls = {}"]:::normal
    end

    subgraph PHASE2 ["Phase 2: RayClassWithInitArgs 包装"]
        direction TB
        WRAP_AR["RayClassWithInitArgs(<br/>ActorRollout Worker,<br/>config, role)"]:::important
        WRAP_CR["RayClassWithInitArgs(<br/>CriticWorker, config)"]:::important
        WRAP_REF["RayClassWithInitArgs(<br/>RefPolicy Worker,<br/>config, role)"]:::normal
    end

    subgraph PHASE3 ["Phase 3: 创建 WorkerGroup"]
        direction TB
        COLOCATE["create_colocated_worker_cls()<br/>=> WorkerDict (代理类)"]:::critical
        WG_INIT["RayWorkerGroup(<br/>resource_pool,<br/>worker_dict_cls)"]:::critical
        SPAWN["wg.spawn(prefix_set)<br/>按角色名拆分 WorkerGroup"]:::important
    end

    subgraph PHASE4 ["Phase 4: 模型初始化"]
        direction TB
        CRITIC_INIT["critic_wg.init_model()"]:::critical
        REF_INIT["ref_policy_wg.init_model()<br/>(如需要)"]:::important
        ACTOR_INIT["actor_rollout_wg.init_model()<br/>🔴 最后初始化<br/>(vLLM 需要估算 KV cache)"]:::critical
    end

    subgraph PHASE5 ["Phase 5: 管理器初始化"]
        direction TB
        REWARD_LOOP["RewardLoopManager(config, pool)<br/>创建 RewardLoopWorker×N"]:::important
        AGENT_LOOP["AgentLoopManager.create()<br/>创建 RolloutReplica,<br/>初始化 LLM Server,<br/>创建 AgentLoopWorker×N"]:::critical
        CKPT_MGR["CheckpointEngineManager(<br/>config, trainer_wg, replicas)"]:::important
        SLEEP["checkpoint_manager<br/>.sleep_replicas()"]:::normal
    end

    IW --> CREATE_RP
    CREATE_RP --> INIT_CLS_MAP
    INIT_CLS_MAP --> WRAP_AR
    WRAP_AR --> WRAP_CR
    WRAP_CR --> WRAP_REF
    WRAP_REF --> COLOCATE
    COLOCATE --> WG_INIT
    WG_INIT --> SPAWN
    SPAWN --> CRITIC_INIT
    CRITIC_INIT --> REF_INIT
    REF_INIT --> ACTOR_INIT
    ACTOR_INIT --> REWARD_LOOP
    REWARD_LOOP --> AGENT_LOOP
    AGENT_LOOP --> CKPT_MGR
    CKPT_MGR --> SLEEP
```

## 7. 核心基础设施层次关系

```mermaid
flowchart BT
    classDef infra fill:#748ffc,stroke:#364fc7,color:#fff
    classDef mid fill:#63e6be,stroke:#087f5b,color:#333
    classDef top fill:#ffa94d,stroke:#e8590c,color:#fff

    subgraph INFRA ["基础设施层"]
        direction LR
        RPM["ResourcePoolManager"]:::infra
        RRP["RayResourcePool<br/>(PlacementGroup)"]:::infra
        PG["Ray PlacementGroup<br/>(GPU 资源绑定)"]:::infra
    end

    subgraph WORKER_LAYER ["Worker 抽象层"]
        direction LR
        RCIA["RayClassWithInitArgs<br/>(类 + 参数打包)"]:::mid
        CCWC["create_colocated_worker_cls<br/>(多角色共享 Actor)"]:::mid
        RWG["RayWorkerGroup<br/>(含 dispatch/collect)"]:::mid
    end

    subgraph MANAGER_LAYER ["管理器层"]
        direction LR
        RLM["RewardLoopManager<br/>(奖励计算)"]:::top
        ALM["AgentLoopManager<br/>(Rollout 调度)"]:::top
        CEM["CheckpointEngineManager<br/>(权重同步)"]:::top
    end

    subgraph TRAINER_LAYER ["训练器层"]
        RPT["RayPPOTrainer<br/>(单控制器编排)"]:::top
    end

    RPM --> RRP
    RRP --> PG
    RCIA --> RWG
    CCWC --> RWG
    RRP --> RWG
    RWG --> RLM
    RWG --> ALM
    RWG --> CEM
    RLM --> RPT
    ALM --> RPT
    CEM --> RPT
```

## 8. 完整数据流向

```mermaid
flowchart LR
    classDef config fill:#da77f2,stroke:#9c36b5,color:#fff
    classDef data fill:#69db7c,stroke:#2b8a3e,color:#fff
    classDef process fill:#4dabf7,stroke:#1971c2,color:#fff
    classDef critical fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:3px

    subgraph INPUT ["输入"]
        YAML["Hydra YAML Config"]:::config
        HF_MODEL["HuggingFace Model<br/>(local/remote)"]:::data
        TRAIN_DATA["训练数据集<br/>(parquet/json)"]:::data
        VAL_DATA["验证数据集"]:::data
    end

    subgraph PROCESS ["处理流"]
        direction TB
        OMCONF["OmegaConf DictConfig"]:::config
        LOCAL_MODEL["Local Model Path"]:::data
        TOKENIZER["Tokenizer"]:::process
        PROCESSOR["Processor (可选)"]:::process
        DATASET["RLHFDataset"]:::data
        RPM["ResourcePoolManager"]:::process
    end

    subgraph OUTPUT ["输出 → RayPPOTrainer"]
        direction TB
        TRAINER["RayPPOTrainer"]:::critical
        WG_ACTOR["actor_rollout_wg"]:::process
        WG_CRITIC["critic_wg"]:::process
        WG_REF["ref_policy_wg"]:::process
        MGR_REWARD["reward_loop_manager"]:::process
        MGR_AGENT["async_rollout_manager"]:::process
        MGR_CKPT["checkpoint_manager"]:::process
        DL_TRAIN["train_dataloader"]:::data
        DL_VAL["val_dataloader"]:::data
    end

    YAML -->|"Hydra 解析"| OMCONF
    HF_MODEL -->|"copy_to_local()"| LOCAL_MODEL
    LOCAL_MODEL --> TOKENIZER
    LOCAL_MODEL --> PROCESSOR
    TRAIN_DATA --> DATASET
    VAL_DATA --> DATASET
    OMCONF --> RPM

    OMCONF --> TRAINER
    TOKENIZER --> TRAINER
    PROCESSOR --> TRAINER
    DATASET --> DL_TRAIN
    DATASET --> DL_VAL
    RPM --> TRAINER

    TRAINER --> WG_ACTOR
    TRAINER --> WG_CRITIC
    TRAINER --> WG_REF
    TRAINER --> MGR_REWARD
    TRAINER --> MGR_AGENT
    TRAINER --> MGR_CKPT
    TRAINER --> DL_TRAIN
    TRAINER --> DL_VAL
```

## 9. `create_colocated_worker_cls` 协同部署机制

> [!NOTE]
> 这是 verl 的核心设计——将多个角色 (Actor/Critic/Ref) 打包到同一个 Ray Actor 中，共享 GPU 资源。

```mermaid
flowchart TD
    classDef wrapper fill:#be4bdb,stroke:#862e9c,color:#fff
    classDef inner fill:#51cf66,stroke:#2b8a3e,color:#fff
    classDef ray fill:#ffa94d,stroke:#e8590c,color:#fff

    INPUT["class_dict = {<br/>'actor_rollout': RayClassWithInitArgs(ActorRolloutCls),<br/>'critic': RayClassWithInitArgs(CriticCls)<br/>}"]:::ray

    DETERMINE["_determine_fsdp_megatron_base_class()<br/>确定基类: Worker / MegatronWorker"]:::wrapper

    subgraph WORKER_DICT ["WorkerDict (动态代理类)"]
        direction TB
        WD_INIT["__init__():<br/>self.worker_dict = {<br/>'actor_rollout': ActorRolloutCls(...),<br/>'critic': CriticCls(...)<br/>}"]:::inner
        WD_METHODS["monkey-patch 方法:<br/>actor_rollout_init_model()<br/>actor_rollout_forward()<br/>critic_init_model()<br/>critic_compute_values()"]:::inner
    end

    RAY_REMOTE["ray.remote(WorkerDict)"]:::ray
    WRAP_RESULT["RayClassWithInitArgs(cls=remote_cls)"]:::ray

    INPUT --> DETERMINE
    DETERMINE --> WD_INIT
    WD_INIT --> WD_METHODS
    WD_METHODS --> RAY_REMOTE
    RAY_REMOTE --> WRAP_RESULT
```

## 10. `AgentLoopManager.create()` 异步初始化

```mermaid
flowchart TD
    classDef critical fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:3px
    classDef important fill:#ffa94d,stroke:#e8590c,color:#fff,stroke-width:2px
    classDef normal fill:#4dabf7,stroke:#1971c2,color:#fff

    CREATE["AgentLoopManager.create()"]:::critical

    INIT_INST["AgentLoopManager.__init__()"]:::normal
    INIT_SERVERS["_initialize_llm_servers()<br/>创建 RolloutReplica×N"]:::critical
    INIT_HYBRID["replica.init_hybrid(worker_group)<br/>🔴 启动 vLLM/SGLang Server"]:::critical
    INIT_LB["_init_global_load_balancer()<br/>GlobalRequestLoadBalancer"]:::normal
    INIT_WORKERS["_init_agent_loop_workers()<br/>AgentLoopWorker×num_workers<br/>(Round-Robin 分布到各节点)"]:::important

    CREATE --> INIT_INST
    INIT_INST --> INIT_SERVERS
    INIT_SERVERS --> INIT_HYBRID
    INIT_HYBRID --> INIT_LB
    INIT_LB --> INIT_WORKERS
```

---

## 关键文件索引

| 组件 | 文件路径 |
|------|---------|
| 入口 main_ppo | [main_ppo.py](file:///home/robomaster/Research/verl/verl/trainer/main_ppo.py) |
| RayPPOTrainer | [ray_trainer.py](file:///home/robomaster/Research/verl/verl/trainer/ppo/ray_trainer.py) |
| Role 枚举 | [utils.py](file:///home/robomaster/Research/verl/verl/trainer/ppo/utils.py) |
| Ray 基础设施 | [base.py](file:///home/robomaster/Research/verl/verl/single_controller/ray/base.py) |
| RewardLoopManager | [reward_loop.py](file:///home/robomaster/Research/verl/verl/experimental/reward_loop/reward_loop.py) |
| AgentLoopManager | [agent_loop.py](file:///home/robomaster/Research/verl/verl/experimental/agent_loop/agent_loop.py) |
| CheckpointEngineManager | [base.py](file:///home/robomaster/Research/verl/verl/checkpoint_engine/base.py) |
| PPO 运行时常量 | [constants_ppo.py](file:///home/robomaster/Research/verl/verl/trainer/constants_ppo.py) |
