import inspect
from typing import Annotated, Any, get_args, get_origin

from nonebot.dependencies import Dependent
from nonebot.internal.params import DependsInner

from .context import (
    Hidden,
    RunContext,
    _InjectMarker,
    _is_run_context_type,
    global_dependency_registry,
)


class BaseParamResolver:
    """可插拔参数解析器基类协议"""

    def match(
        self,
        name: str,
        param: inspect.Parameter,
        context: RunContext,
        available_injects: dict[str, Any],
        inject_kwargs: dict[str, Any],
    ) -> bool:
        return False

    def static_match(self, param: inspect.Parameter) -> bool:
        return False

    async def resolve(
        self,
        name: str,
        param: inspect.Parameter,
        context: RunContext,
        available_injects: dict[str, Any],
        inject_kwargs: dict[str, Any],
    ) -> Any:
        raise NotImplementedError


class RunContextResolver(BaseParamResolver):
    def match(self, name, param, context, available_injects, inject_kwargs) -> bool:
        return _is_run_context_type(param.annotation)

    def static_match(self, param: inspect.Parameter) -> bool:
        return _is_run_context_type(param.annotation)

    async def resolve(
        self, name, param, context, available_injects, inject_kwargs
    ) -> Any:
        return context


class DepsResolver(BaseParamResolver):
    def match(self, name, param, context, available_injects, inject_kwargs) -> bool:
        if context.deps is None:
            return False
        actual_type = param.annotation
        if get_origin(actual_type) is Annotated:
            actual_type = get_args(actual_type)[0]
        try:
            return isinstance(context.deps, actual_type)
        except TypeError:
            return False

    def static_match(self, param: inspect.Parameter) -> bool:
        return False

    async def resolve(
        self, name, param, context, available_injects, inject_kwargs
    ) -> Any:
        return context.deps


class ScopeContextResolver(BaseParamResolver):
    def match(self, name, param, context, available_injects, inject_kwargs) -> bool:
        if name in available_injects:
            return True

        actual_type = param.annotation
        if get_origin(actual_type) is Annotated:
            actual_type = get_args(actual_type)[0]

        if actual_type is inspect.Parameter.empty or actual_type is Any:
            return False

        for val in available_injects.values():
            origin_type = get_origin(actual_type) or actual_type
            try:
                if isinstance(val, origin_type):
                    return True
            except TypeError:
                pass
        return False

    def static_match(self, param: inspect.Parameter) -> bool:
        actual_type = param.annotation
        if get_origin(actual_type) is Annotated:
            actual_type = get_args(actual_type)[0]
        anno_str = str(actual_type)
        return any(
            kw in anno_str
            for kw in ("Bot", "Event", "EventSession", "Uninfo", "Matcher")
        )

    async def resolve(
        self, name, param, context, available_injects, inject_kwargs
    ) -> Any:
        if name in available_injects:
            return available_injects[name]

        actual_type = param.annotation
        if get_origin(actual_type) is Annotated:
            actual_type = get_args(actual_type)[0]

        for val in available_injects.values():
            origin_type = get_origin(actual_type) or actual_type
            try:
                if isinstance(val, origin_type):
                    return val
            except TypeError:
                pass

        raise ValueError(f"无法解析依赖: 未找到类型为 {actual_type} 的实例")


class ExtraStateResolver(BaseParamResolver):
    def match(self, name, param, context, available_injects, inject_kwargs) -> bool:
        actual_type = param.annotation
        if get_origin(actual_type) is Annotated:
            actual_type = get_args(actual_type)[0]
        type_name = getattr(actual_type, "__name__", "")
        di_key = f"di_type_{type_name}"
        return di_key in getattr(context, "extra", {})

    async def resolve(
        self, name, param, context, available_injects, inject_kwargs
    ) -> Any:
        actual_type = param.annotation
        if get_origin(actual_type) is Annotated:
            actual_type = get_args(actual_type)[0]
        type_name = getattr(actual_type, "__name__", "")
        di_key = f"di_type_{type_name}"
        return context.extra[di_key]


class GlobalRegistryResolver(BaseParamResolver):
    def match(self, name, param, context, available_injects, inject_kwargs) -> bool:
        actual_type = param.annotation
        if get_origin(actual_type) is Annotated:
            actual_type = get_args(actual_type)[0]
        return global_dependency_registry.get(actual_type) is not None

    def static_match(self, param: inspect.Parameter) -> bool:
        actual_type = param.annotation
        if get_origin(actual_type) is Annotated:
            actual_type = get_args(actual_type)[0]
        return global_dependency_registry.has_provider(actual_type)

    async def resolve(
        self, name, param, context, available_injects, inject_kwargs
    ) -> Any:
        actual_type = param.annotation
        if get_origin(actual_type) is Annotated:
            actual_type = get_args(actual_type)[0]
        return global_dependency_registry.get(actual_type)


class HiddenInjectsResolver(BaseParamResolver):
    def match(self, name, param, context, available_injects, inject_kwargs) -> bool:
        is_hidden = False
        if get_origin(param.annotation) is Annotated:
            args = get_args(param.annotation)
            if any(isinstance(arg, Hidden) or arg is Hidden for arg in args):
                is_hidden = True
        return is_hidden and name in inject_kwargs

    def static_match(self, param: inspect.Parameter) -> bool:
        if get_origin(param.annotation) is Annotated:
            args = get_args(param.annotation)
            if any(isinstance(arg, Hidden) or arg is Hidden for arg in args):
                return True
        return False

    async def resolve(
        self, name, param, context, available_injects, inject_kwargs
    ) -> Any:
        return inject_kwargs[name]


class NonebotDependsResolver(BaseParamResolver):
    def match(self, name, param, context, available_injects, inject_kwargs) -> bool:
        if isinstance(param.default, DependsInner):
            return True
        if get_origin(param.annotation) is Annotated:
            for arg in get_args(param.annotation):
                if isinstance(arg, DependsInner):
                    return True
        return False

    def static_match(self, param: inspect.Parameter) -> bool:
        if isinstance(param.default, DependsInner):
            return True
        if get_origin(param.annotation) is Annotated:
            for arg in get_args(param.annotation):
                if isinstance(arg, DependsInner):
                    return True
        return False

    async def resolve(
        self, name, param, context, available_injects, inject_kwargs
    ) -> Any:
        depends_inner = None
        if isinstance(param.default, DependsInner):
            depends_inner = param.default
        elif get_origin(param.annotation) is Annotated:
            for arg in get_args(param.annotation):
                if isinstance(arg, DependsInner):
                    depends_inner = arg
                    break

        dependency_func = depends_inner.dependency if depends_inner else None
        if dependency_func is None:
            dependency_func = param.annotation

        dependent = Dependent[Any].parse(
            call=dependency_func, parameterless=(), allow_types=set()
        )
        try:
            return await dependent(**inject_kwargs)
        except Exception as e:
            raise ValueError(f"解析依赖 {name} 失败: {e}")


class TypeSugarResolver(BaseParamResolver):
    def _get_marker(self, param: inspect.Parameter) -> str | None:
        anno = param.annotation
        if hasattr(anno, "__metadata__"):
            for arg in anno.__metadata__:
                if isinstance(arg, _InjectMarker):
                    return arg.key
        return None

    def match(self, name, param, context, available_injects, inject_kwargs) -> bool:
        return self._get_marker(param) is not None

    def static_match(self, param: inspect.Parameter) -> bool:
        return self._get_marker(param) is not None

    async def resolve(
        self, name, param, context, available_injects, inject_kwargs
    ) -> Any:
        marker = self._get_marker(param)
        if marker == "user_id":
            return context.get_user_id()
        elif marker == "group_id":
            return context.get_group_id()
        elif marker == "platform":
            return context.get_platform()
        elif marker == "bot":
            return context.bot
        elif marker == "event":
            return context.event
        elif marker == "matcher":
            return context.matcher
        elif marker == "session":
            if context.bot and context.event:
                from nonebot_plugin_session import extract_session

                return extract_session(context.bot, context.event)
            return None
        raise ValueError(f"未知的类型糖标记: {marker}")


class DependencyInjector:
    """可插拔依赖注入管线 (Resolver Pipeline)"""

    _resolvers: list[BaseParamResolver] = []

    @classmethod
    def register(cls, resolver: BaseParamResolver) -> None:
        cls._resolvers.append(resolver)

    @classmethod
    def can_resolve_statically(cls, param: inspect.Parameter) -> bool:
        for resolver in cls._resolvers:
            if resolver.static_match(param):
                return True
        return False

    @classmethod
    async def resolve_all(
        cls,
        sig: inspect.Signature,
        call_kwargs: dict[str, Any],
        inject_kwargs: dict[str, Any],
        context: RunContext,
        available_injects: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        for name, param in sig.parameters.items():
            if name in ("self", "cls") or name in call_kwargs:
                continue

            for resolver in cls._resolvers:
                if resolver.match(
                    name, param, context, available_injects, inject_kwargs
                ):
                    val = await resolver.resolve(
                        name, param, context, available_injects, inject_kwargs
                    )
                    call_kwargs[name] = val
                    inject_kwargs[name] = val
                    break

        return call_kwargs, inject_kwargs


DependencyInjector.register(RunContextResolver())
DependencyInjector.register(DepsResolver())
DependencyInjector.register(TypeSugarResolver())
DependencyInjector.register(ScopeContextResolver())
DependencyInjector.register(ExtraStateResolver())
DependencyInjector.register(GlobalRegistryResolver())
DependencyInjector.register(HiddenInjectsResolver())
DependencyInjector.register(NonebotDependsResolver())
