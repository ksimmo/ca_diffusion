def init_flags(cfg):
    global USE_DEEPSPEED
    USE_DEEPSPEED = cfg.get("use_deepspeed", False)

    global FORCE_CHECKPOINTING
    FORCE_CHECKPOINTING = cfg.get("force_checkpointing", False)
