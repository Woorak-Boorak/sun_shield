const pattern = /\((\d+),(\d+),(\d+)\)/; // 받은 메시지에서 특정 값 추출을 위한 정규식
let write = true; // 그릴지 여부
const circle = document.querySelector("div");

// 웹소켓 연결
const webSocket = new WebSocket("ws://localhost:8000");

webSocket.onopen = function () {
  console.log("Web Socket Connected");
  webSocket.send("안녕하세요"); // 서버에 메시지 보내보기
};

webSocket.onmessage = function (message) {
  console.log(message.data); // 콘솔 표시
  if (pattern.test(message.data)) {
    console.log("ok");
    const info = pattern.exec(message.data);
    console.log(typeof info[0]);
    console.log(info[1]);
    if (info[1] == "0") {
      circle.id = "";
      write = false;
      console.log("안보임");
    } else {
      circle.id = "visality";
      write = true;
      console.log("보임");
    }
    console.log("y:", info[2] + "%");
    circle.style.top = info[2] + "%";
    console.log("x:", info[3] + "%");
    circle.style.left = info[3] + "%";
  }
};
