from agent.config import parse_env


def test_env_inline_comments_and_quotes():
    e = parse_env("""# c
AGENT_THINK=false          # no reasoning
AGENT_WORKSPACE=~/Projects # sandbox root
A="x # not a comment"
B='y'
C=http://h:1/p#frag
""")
    assert e == {"AGENT_THINK": "false", "AGENT_WORKSPACE": "~/Projects", "A": "x # not a comment", "B": "y",
                 "C": "http://h:1/p#frag"}
