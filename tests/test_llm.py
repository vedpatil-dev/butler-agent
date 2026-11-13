import json

import pytest

pytestmark = pytest.mark.llm


def test_basic_generation(llm):
    r = llm.complete([{"role": "user", "content": "What is the capital of France? One word."}], num_predict=20)
    assert "paris" in r["content"].lower() and r["stats"]["gen_tps"] > 0


def test_coding(llm):
    r = llm.complete([{"role": "user", "content": "Write a Java method isPalindrome(String s). Code only."}], num_predict=300)
    assert "isPalindrome" in r["content"] and "boolean" in r["content"]


def test_json_output(llm):
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"}},
              "required": ["city", "population_millions"]}
    r = llm.complete([{"role": "user", "content": "Give one large city in Japan with its approximate population."}],
                     fmt=schema, num_predict=80)
    d = json.loads(r["content"])
    assert isinstance(d["city"], str) and isinstance(d["population_millions"], (int, float))


def test_tool_calling(llm):
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    r = llm.complete([{"role": "user", "content": "What's the weather in Paris? Use the tool."}], tools=tools, num_predict=100)
    c = r["tool_calls"][0]["function"]
    assert c["name"] == "get_weather" and "paris" in json.dumps(c["arguments"]).lower()


def test_streaming(llm):
    kinds = [k for k, _ in llm.stream([{"role": "user", "content": "Count from 1 to 5."}], num_predict=40)]
    assert kinds.count("text") > 1 and kinds[-1] == "stats"
