"""Gen ubee main."""
# pylint: disable=unused-import, wrong-import-position, wrong-import-order, too-many-locals, broad-except

from typing import Tuple, Optional

from pathlib import Path
import sys
from random import shuffle

from itertools import zip_longest
from textwrap import dedent

import gradio as gr

import pandas as pd
from icecream import install as ic_install, ic
import logzero
from logzero import logger

# for embeddable python
if "." not in sys.path:
    sys.path.insert(0, ".")

from ubee.ubee import ubee

# logzero.loglevel(10)
ic_install()
ic.configureOutput(
    includeContext=True,
    outputFunction=logger.info,
)
ic.enable()
# ic.disenable()  # to turn off


def greet1(name):
    """Dummy."""
    return "Hello " + name + "!!"


def greet(
    text1,
    text2,
    # segment: str
    thresh: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Take inputs, return outputs.

    Args:
        text1: text
        text2: text
    Returns:
        pd.DataFrame
    """
    res1 = [elm.strip() for elm in text1.splitlines() if elm.strip()]
    res2 = [elm.strip() for elm in text2.splitlines() if elm.strip()]

    ic(res1)
    ic(res2)

    # _ = pd.DataFrame(zip_longest(res1, res2), columns=["text1", "text2"])
    # return _

    res1_, res2_ = ubee(res1, res2, thresh)

    out_df = pd.DataFrame(
        zip_longest(res1, res2),
        columns=["text1", "text2"],
    )

    if res2_:
        _ = pd.DataFrame(res2_, columns=["text1", "text2"])
    else:
        _ = None

    return out_df, pd.DataFrame(res1_, columns=["text1", "text2", "likelihood"]), _


def main():
    """Create main entry."""
    text_zh = Path("data/test_zh.txt").read_text(encoding="utf8")
    text_zh = [elm.strip() for elm in text_zh.splitlines() if elm.strip()][:10]
    text_zh = "\n\n".join(text_zh)

    text_en = [
        elm.strip()
        for elm in Path("data/test_en.txt").read_text(encoding="utf8").splitlines()
        if elm.strip()
    ]
    _ = text_en[:9]
    shuffle(_)
    text_en = "\n\n".join(_)

    title = "Ultimatumbee Aligner"
    theme = "dark-grass"
    theme = "grass"
    description = """WIP showcasing a novel aligner"""
    article = dedent("""
        ## NB

        *   The ultimatumbee aligner (``ubee`` for short) is intended for aligning text blocks (be it paragraphs, sentences or words). Since it is rather slow (30 para pairs (Wuthering Height ch1. for example) can take 10 to 20 mniutes), anything more than 50 blocks should probably be avaoided. Nevertheless, you are welcome to try. No big brother is watching.

        *   ``thresh``: longer text blocks justify a larger value; `.5` appears to be just right for paragraphs for Wuthering Height ch1.

        Stay tuned for more details coming soon...
        """).strip()

    ex1_zh = [
        '雪开始下大了。',
        '我握住门柄又试一回。',
        '这时一个没穿外衣的年轻人，扛着一根草耙，在后面院子里出现了。',        '他招呼我跟着他走，穿过了一个洗衣房和一片铺平的地，那儿有煤棚、抽水机和鸽笼，我们终于到了我上次被接待过的那间温暖的、热闹的大屋子。',
        '煤、炭和木材混合在一起燃起的熊熊炉火，使这屋子放着光彩。', '在准备摆上丰盛晚餐的桌旁，我很高兴地看到了那位“太太”，以前我从未料想到会有这么一个人存在的。',
        '我鞠躬等候，以为她会叫我坐下。',
        '她望望我，往她的椅背一靠，不动，也不出声。'
    ]
    ex1_en = [
        'The snow began to drive thickly.',
         'I seized the handle to essay another trial; when a young man without coat, and shouldering a pitchfork, appeared in the yard behind.',
         'He hailed me to follow him, and, after marching through a wash-house, and a paved area containing a coal shed, pump, and pigeon cot, we at length arrived in the huge, warm, cheerful apartment, where I was formerly received.',
         "It glowed delightfully in the radiance of an immense fire, compounded of coal, peat, and wood; and near the table, laid for a plentiful evening meal, I was pleased to observe the `missis', an individual whose existence I had never previously suspected.",
         'I bowed and waited, thinking she would bid me take a seat.',
         'She looked at me, leaning back in her chair, and remained motionless and mute.'
    ]
    shuffle(ex1_en)
    ex1_zh = "\n".join(ex1_zh)
    ex1_en = "\n".join(ex1_en)

    ex2_zh = "她\n望望\n我\n往\n她\n的\n椅背\n一靠\n不\n动\n也\n不\n出声"
    ex2_en = "She looked at me leaning back in her chair and remained motionless and mute".split()
    shuffle(ex2_en)
    ex2_en = "\n".join(ex2_en)

    examples = [
        [ex2_zh, ex2_en, .3],
        [text_zh, text_en, .5],
    ]
    lines = 15
    placeholder = "Type or paste text here"

    inputs = [
        gr.inputs.Textbox(
            lines=lines, placeholder=placeholder, default=ex1_zh, label="text1"
        ),
        gr.inputs.Textbox(
            lines=lines, placeholder=placeholder, default=ex1_en, label="text2"
        ),
        gr.inputs.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.1,
            default=0.4,
            label="threshold",
        ),
    ]

    out_df = gr.outputs.Dataframe(
        headers=None,
        max_rows=lines,  # 20
        max_cols=None,
        overflow_row_behaviour="paginate",
        type="auto",
        label="To be aligned",
    )
    aligned = gr.outputs.Dataframe(
        headers=None,
        max_rows=lines,  # 20
        max_cols=None,
        overflow_row_behaviour="paginate",
        type="auto",
        label="Aligned",
    )
    leftover = gr.outputs.Dataframe(
        headers=None,
        max_rows=lines,  # 20
        max_cols=None,
        overflow_row_behaviour="paginate",
        type="auto",
        label="Leftover",
    )
    outputs = [  # tot. 3
        out_df,
        aligned,
        leftover,
    ]

    iface = gr.Interface(
        fn=greet,
        # fn=ubee,
        title=title,
        theme=theme,
        layout="vertical",  # horizontal unaligned
        description=description,
        article=article,
        # inputs="text",
        # outputs="text",
        inputs=inputs,  # text1, text2, segment, thresh
        outputs=outputs,
        examples=examples,
        # enable_queue=True,
    )
    iface.launch(
        enable_queue=True,
        share=True,
    )


if __name__ == "__main__":
    main()

_ = """

        gr.inputs.Radio(
            ["para", "sent", "word"],
            default="para",
            label="segment"
        )
# """