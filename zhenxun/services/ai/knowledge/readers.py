import asyncio
import csv
from pathlib import Path
from urllib.parse import urlparse

from zhenxun.services.ai.knowledge.models import Document
from zhenxun.services.log import logger
from zhenxun.utils.http_utils import AsyncHttpx
from zhenxun.utils.user_agent import get_user_agent


class BaseReader:
    """读取器基类"""

    def read(self, file_path: Path) -> Document | None:
        raise NotImplementedError


class TextReader(BaseReader):
    """处理 .txt, .md, .json 等纯文本"""

    def read(self, file_path: Path) -> Document | None:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return Document(name=file_path.name, content=content)
        except Exception as e:
            logger.error(f"[TextReader] 读取文件失败 {file_path}: {e}")
            return None


class CSVReader(BaseReader):
    """
    处理 .csv 报表
    将其标准化为逗号分隔的文本块，便于后续的 RowChunking 处理。
    """

    def read(self, file_path: Path) -> Document | None:
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                lines = []
                for row in reader:
                    clean_row = [
                        str(cell).replace("\n", " ").replace("\r", "") for cell in row
                    ]
                    lines.append(",".join(clean_row))

            content = "\n".join(lines)
            return Document(name=file_path.name, content=content)
        except Exception as e:
            logger.error(f"[CSVReader] 读取 CSV 失败 {file_path}: {e}")
            return None


def _clean_html_sync(html_content: str) -> str:
    """[CPU密集型任务] 清洗 HTML，提取纯文本，必须放入线程池运行"""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "html.parser")

        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        text = soup.get_text(separator=" ", strip=True)
        import re

        text = re.sub(r"\s+", " ", text)
        return text
    except Exception as e:
        logger.warning(f"HTML 解析失败: {e}")
        return html_content


class WebReader(BaseReader):
    """处理网页 URL (无阻塞架构)"""

    def __init__(self, timeout: int = 15, max_length: int = 20000):
        self.timeout = timeout
        self.max_length = max_length

    async def read_async(self, url: str) -> Document | None:
        try:
            headers = get_user_agent()
            response = await AsyncHttpx.get(
                url,
                headers=headers,
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
            html_content = response.text

            text_content = await asyncio.to_thread(_clean_html_sync, html_content)

            if len(text_content) > self.max_length:
                text_content = text_content[: self.max_length] + "\n...[内容过长已截断]"

            return Document(
                name=urlparse(url).netloc,
                content=text_content,
                meta_data={"source": "web_reader", "url": url},
            )
        except Exception as e:
            logger.error(f"[WebReader] 读取网页失败 {url}: {e}")
            return None


def get_reader_for_file(file_path: Path) -> BaseReader | None:
    """工厂方法：根据文件后缀自动分配读取器"""
    ext = file_path.suffix.lower()

    if ext in [".txt", ".md", ".json", ".log", ".yaml", ".yml", ".ini"]:
        return TextReader()
    elif ext in [".csv"]:
        return CSVReader()

    logger.warning(f"暂不支持解析文件后缀: {ext}")
    return None


def get_reader_for_url(url: str) -> WebReader | None:
    """工厂方法：获取 URL 读取器"""
    if url.startswith(("http://", "https://")):
        return WebReader()
    return None

