from tortoise import fields

from zhenxun.services.db_context import Model


class PluginLimitState(Model):
    id = fields.IntField(pk=True, generated=True)
    scope = fields.CharField(max_length=32, description="作用域: USER, GROUP, GLOBAL")
    subject_id = fields.CharField(
        max_length=255, description="对象ID: user_id, group_id"
    )
    plugin = fields.CharField(max_length=255, description="插件模块名")
    node = fields.CharField(max_length=255, description="限制节点标识")

    state = fields.JSONField(default={}, description="限制器状态数据")
    expire_at = fields.FloatField(index=True, description="绝对过期时间戳")

    class Meta:  # type: ignore
        table = "plugin_limit_states"
        table_description = "插件限制器状态表"
        unique_together = (("scope", "subject_id", "plugin", "node"),)
