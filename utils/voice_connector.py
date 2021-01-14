import base64
import time
import uuid
from io import BytesIO
from typing import Any, Awaitable, Callable, Dict, Optional, Text

import librosa
from components.deepspeech import DeepSpeechModel
from components.tts import tts_run
from rasa.core.channels import SocketIOInput, UserMessage
from rasa.core.channels.socketio import SocketBlueprint, SocketIOOutput
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from socketio import AsyncServer

ds = DeepSpeechModel()


class VoiceOutput(SocketIOOutput):
    FILE_SERVER = "http://localhost:8888/"

    async def send_text_message(
            self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        """Send a message through this channel."""
        print("开始发送信息")
        await self._send_audio_message(socket_id=recipient_id, response={"text": text})

    async def _send_audio_message(self, socket_id: str, response: Any) -> None:
        """Sends a message to the recipient using the bot event."""

        ts = time.time()
        out_file_name = str(ts) + ".wav"
        link = self.FILE_SERVER + out_file_name
        tts_run(text=response["text"], file_name=out_file_name)  # TTS 生成语音文件
        await self._send_message(
            response={"text": response["text"], "link": link}, socket_id=socket_id
        )  # 发送给前端


class VoiceInput(SocketIOInput):
    def blueprint(
            self, on_new_message: Callable[[UserMessage], Awaitable[Any]]
    ) -> Blueprint:
        # Workaround so that socketio works with requests from other origins.
        # https://github.com/miguelgrinberg/python-socketio/issues/205#issuecomment-493769183
        sio = AsyncServer(async_mode="sanic", cors_allowed_origins="*")
        socketio_webhook = SocketBlueprint(
            sio, self.socketio_path, "socketio_webhook", __name__
        )

        # make sio object static to use in get_output_channel
        self.sio = sio

        @socketio_webhook.route("/", methods=["GET"])
        async def health(_: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @sio.on("connect", namespace=self.namespace)
        async def connect(sid: Text, _) -> None:
            print(f"User {sid} connected to socketIO endpoint.")

        @sio.on("disconnect", namespace=self.namespace)
        async def disconnect(sid: Text) -> None:
            print(f"User {sid} disconnected from socketIO endpoint.")

        @sio.on("session_request", namespace=self.namespace)
        async def session_request(sid: Text, data: Optional[Dict]):
            if data is None:
                data = {}
            if "session_id" not in data or data["session_id"] is None:
                data["session_id"] = uuid.uuid4().hex
            if self.session_persistence:
                sio.enter_room(sid, data["session_id"])
            await sio.emit("session_confirm", data["session_id"], room=sid)
            print(f"User {sid} connected to socketIO endpoint.")

        @sio.on(self.user_message_evt, namespace=self.namespace)
        async def handle_message(sid: Text, data: Dict) -> Any:
            """处理收到的客户端数据"""
            output_channel = VoiceOutput(sio, self.bot_message_evt)

            message = data["message"]
            if message == "/get_started":
                message = data["message"]
            else:
                bytes_data = base64.b64decode(message.split(",", maxsplit=1)[-1])  # 解码 base64 获取语音元数据
                audio, fs = librosa.load(
                    BytesIO(bytes_data), sr=None, dtype="int16", mono=False
                )  # 获取声波数据
                print("获取音频信息成功")
                message = ds.predict_to_string(audio, fs)  # STT
                print(f"预测输出为: {message}")

            message = UserMessage(
                message, output_channel, sid, input_channel=self.name()
            )
            await on_new_message(message)

        return socketio_webhook
