import os
import sys
import json
import logging
import time

import openai
import tokentrim as tt

from .run_text_llm import run_text_llm
from .run_tool_calling_llm import run_tool_calling_llm
from .utils.convert_to_openai_messages import convert_to_openai_messages


def _supports_function_calling(model: str) -> bool:
    base = model.split("/")[-1]
    # Older completions-only models that predate tool/function calling
    _no_tools = {"text-davinci-003", "text-davinci-002", "code-davinci-002"}
    return base not in _no_tools


def _supports_vision(model: str) -> bool:
    base = model.split("/")[-1]
    openai_vision = any(k in base for k in ("4o", "4-turbo", "vision", "gpt-4v"))
    claude_vision = base.startswith("claude-3")
    return openai_vision or claude_vision


class Llm:
    """
    A stateless LMC-style LLM with some helpful properties.
    """

    def __init__(self, interpreter):
        self.interpreter = interpreter

        self.completions = openai_completions

        self.model = "gpt-4o"
        self.temperature = 0.0

        self.supports_vision = None
        self.vision_renderer = self.interpreter.computer.vision.query

        self.supports_functions = None
        self.execution_instructions = "To execute code on the user's machine, write a markdown code block. Specify the language after the ```. You will receive the output. Use any programming language."

        self.context_window = None
        self.max_tokens = None
        self.api_base = None
        self.api_key = None
        self.api_version = None
        self._is_loaded = False

    def run(self, messages):
        if not self._is_loaded:
            self.load()

        if (
            self.max_tokens is not None
            and self.context_window is not None
            and self.max_tokens > self.context_window
        ):
            print(
                "Warning: max_tokens is larger than context_window. Setting max_tokens to be 0.2 times the context_window."
            )
            self.max_tokens = int(0.2 * self.context_window)

        assert messages[0]["role"] == "system", "First message must have the role 'system'"
        for msg in messages[1:]:
            assert msg["role"] != "system", "No message after the first can have the role 'system'"

        model = self.model
        if model in ["claude-3.5", "claude-3-5", "claude-3.5-sonnet", "claude-3-5-sonnet"]:
            model = "claude-3-5-sonnet-20240620"
            self.model = "claude-3-5-sonnet-20240620"

        if self.supports_functions is None:
            self.supports_functions = _supports_function_calling(model)

        if self.supports_vision is None:
            self.supports_vision = _supports_vision(model)

        # Trim image messages
        image_messages = [msg for msg in messages if msg["type"] == "image"]
        if self.supports_vision:
            if self.interpreter.os:
                if len(image_messages) > 1:
                    for img_msg in image_messages[:-2]:
                        messages.remove(img_msg)
                        if self.interpreter.verbose:
                            print("Removing image message!")
            else:
                if len(image_messages) > 3:
                    for img_msg in image_messages[1:-2]:
                        messages.remove(img_msg)
                        if self.interpreter.verbose:
                            print("Removing image message!")
        elif self.supports_vision == False and self.vision_renderer:
            for img_msg in image_messages:
                if img_msg["format"] != "description":
                    self.interpreter.display_message("\n  *Viewing image...*\n")

                    if img_msg["format"] == "path":
                        precursor = f"The image I'm referring to ({img_msg['content']}) contains the following: "
                        if self.interpreter.computer.import_computer_api:
                            postcursor = f"\nIf you want to ask questions about the image, run `computer.vision.query(path='{img_msg['content']}', query='(ask any question here)')` and a vision AI will answer it."
                        else:
                            postcursor = ""
                    else:
                        precursor = "Imagine I have just shown you an image with this description: "
                        postcursor = ""

                    try:
                        image_description = self.vision_renderer(lmc=img_msg)
                        ocr = self.interpreter.computer.vision.ocr(lmc=img_msg)
                        img_msg["content"] = (
                            precursor
                            + image_description
                            + "\n---\nI've OCR'd the image, this is the result (this may or may not be relevant. If it's not relevant, ignore this): '''\n"
                            + ocr
                            + "\n'''"
                            + postcursor
                        )
                        img_msg["format"] = "description"
                    except ImportError:
                        print("\nTo use local vision, run `pip install 'open-interpreter[local]'`.\n")
                        img_msg["format"] = "description"
                        img_msg["content"] = ""

        messages = convert_to_openai_messages(
            messages,
            function_calling=self.supports_functions,
            vision=self.supports_vision,
            shrink_images=self.interpreter.shrink_images,
            interpreter=self.interpreter,
        )

        system_message = messages[0]["content"]
        messages = messages[1:]

        try:
            if self.context_window and self.max_tokens:
                trim_to_be_this_many_tokens = self.context_window - self.max_tokens - 25
                messages = tt.trim(
                    messages,
                    system_message=system_message,
                    max_tokens=trim_to_be_this_many_tokens,
                )
            elif self.context_window and not self.max_tokens:
                messages = tt.trim(
                    messages,
                    system_message=system_message,
                    max_tokens=self.context_window,
                )
            else:
                try:
                    messages = tt.trim(messages, system_message=system_message, model=model)
                except:
                    if len(messages) == 1:
                        if self.interpreter.in_terminal_interface:
                            self.interpreter.display_message(
                                """
**We were unable to determine the context window of this model.** Defaulting to 8000.

If your model can handle more, run `interpreter --context_window {token limit} --max_tokens {max tokens per response}`.

Continuing...
                            """
                            )
                        else:
                            self.interpreter.display_message(
                                """
**We were unable to determine the context window of this model.** Defaulting to 8000.

If your model can handle more, run `self.context_window = {token limit}`.

Also please set `self.max_tokens = {max tokens per response}`.

Continuing...
                            """
                            )
                    messages = tt.trim(messages, system_message=system_message, max_tokens=8000)
        except:
            messages = [{"role": "system", "content": system_message}] + messages

        if system_message == "":
            if messages[0]["role"] != "system":
                messages = [{"role": "system", "content": system_message}] + messages

        params = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        if self.api_key:
            params["api_key"] = self.api_key
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_version:
            params["api_version"] = self.api_version
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if self.temperature:
            params["temperature"] = self.temperature
        if hasattr(self.interpreter, "conversation_id"):
            params["conversation_id"] = self.interpreter.conversation_id

        if self.interpreter.verbose:
            logging.basicConfig(level=logging.DEBUG)

        if self.supports_functions:
            yield from run_tool_calling_llm(self, params)
        else:
            yield from run_text_llm(self, params)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self._is_loaded = False

    def load(self):
        if self._is_loaded:
            return
        self._is_loaded = True

        if self.context_window is None:
            self.context_window = 8192
        if self.max_tokens is None:
            self.max_tokens = int(self.context_window * 0.2)


def openai_completions(**params):
    api_key = params.pop("api_key", None) or os.environ.get("OPENAI_API_KEY") or "x"
    api_base = params.pop("api_base", None)
    params.pop("api_version", None)
    params.pop("conversation_id", None)

    params["model"] = params["model"].replace(":latest", "").replace("openai/", "")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_base or None,
        timeout=60.0,
    )

    attempts = 4
    first_error = None

    for attempt in range(attempts):
        try:
            yield from client.chat.completions.create(**params)
            return
        except KeyboardInterrupt:
            print("Exiting...")
            sys.exit(0)
        except Exception as e:
            if attempt == 0:
                first_error = e
            if isinstance(e, openai.AuthenticationError) and api_key in ("x", None):
                print(
                    "OpenAI requires an API key. Trying again with a dummy API key. In the future, if this fixes it, please set a dummy API key to prevent this message. (e.g `interpreter --api_key x` or `self.api_key = 'x'`)"
                )
                client = openai.OpenAI(api_key="x", base_url=api_base or None)
            if attempt == 1:
                params["temperature"] = params.get("temperature", 0.0) + 0.1

    if first_error is not None:
        raise first_error
