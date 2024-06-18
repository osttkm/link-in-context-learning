_base_ = [
    'DEFAULT_VAL_PG.py',
    'DEFAULT_VAL_VI.py',
]

DEFAULT_TRAIN_DATASET = dict(
    custom_val_vi_data = dict(
    **_base_.DEFAULT_VAL_VI_CUSTOMDATA_VARIANT.customdata_train,
    policy="policy_2way_weight"
    ),
    custom_val_pg_data = dict(
    **_base_.DEFAULT_VAL_PG_CUSTOMDATA_VARIANT.customdata_train,
    policy="policy_2way_weight"
    ),
)
