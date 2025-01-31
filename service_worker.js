chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "message") {
        // Store the message so the popup can access it
        chrome.storage.session.set({ contentMessage: message.data });
        console.log("Got message data: ", message);
    }
});
