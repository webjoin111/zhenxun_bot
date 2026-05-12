from typing import Any

from jinja2 import Template

from zhenxun.services.log import logger


class PromptTemplate:
    """
    核心 Prompt 渲染引擎。
    基于 Jinja2 提供强大的模板变量替换功能。
    """

    def __init__(self, template_string: str):
        self.template_string = template_string

    def render(self, **variables: Any) -> str:
        if not self.template_string or not variables:
            return self.template_string

        try:
            template = Template(self.template_string)
            rendered_string = template.render(**variables)
            logger.debug(f"模板渲染成功: {rendered_string}")
            return rendered_string
        except Exception as e:
            logger.warning(f"Jinja2 模板渲染失败: {e}, 模板原内容: {self.template_string}", e=e)
            return self.template_string
