// --- DOM Elements ---
const chatBox = document.getElementById("chat-box");
const micButton = document.getElementById("mic-button");

// --- Add Messages to Chat ---
eel.expose(addUserMsg);
function addUserMsg(msg) {
    chatBox.innerHTML += `<div class="user-msg">${msg}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
}

eel.expose(addAppMsg);
function addAppMsg(msg) {
    chatBox.innerHTML += `<div class="app-msg">${msg}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
}

// --- Mic Button Click Event ---
micButton.addEventListener("click", () => {
    eel.mic_triggered();
});

// --- Trigger Greeting When Page Loads ---
window.onload = function () {
    eel.greet_user();
};
