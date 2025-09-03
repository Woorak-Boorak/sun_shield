// javascript/script.js
const pattern = /\((\d+),(\d+),(\d+)\)/;
let write = true;
const circle = document.querySelector(".circle"); // ← div 대신 .circle로

const webSocket = new WebSocket("ws://localhost:8000");

webSocket.onopen = function () {
  console.log("Web Socket Connected");
};

webSocket.onmessage = function (message) {
  const s = String(message.data || "").trim();
  const m = pattern.exec(s);
  if (!m) return;

  const v = m[1] === "0" ? 0 : 1;
  const y = parseInt(m[2], 10);
  const x = parseInt(m[3], 10);

  if (v === 0) {
    circle.id = "";
    write = false;
  } else {
    circle.id = "visality";
    write = true;
  }
  circle.style.top = y + "%";
  circle.style.left = x + "%";
};
