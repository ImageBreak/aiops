from llama_index.core import PromptTemplate


QA_TEMPLATE = """\
    上下文信息如下：
    ----------
    {context_str}
    ----------
    请你基于上下文信息而不是自己的知识，回答以下问题，可以分点作答，回答请尽可能简洁明了，不要给人是参考了上下文的感觉，要让人觉得那就是你自己的知识。并且回答前请仔细思考将要回答的内容是否满足提问的真实意图，如果上下文信息没有相关知识，就回答不确定：
    {query_str}

    回答：\
    """


SUMMARY_EXTRACT_TEMPLATE = """\
    这是这一小节的内容：
    {context_str}
    请用中文总结本节的关键主题和实体。

    总结：\
    """
