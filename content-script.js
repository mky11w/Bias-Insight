const getContent = () => {
    return new Promise((res, rej) => {
        // Query for all the content within the page
        const content = (document.body.innerText).split("\n").filter(e => {
            const len = e.split(" ").length;
            return len > 3;
        }).join("\n");

        // Once we get content, send it off to Makayla's app
        fetch("http://localhost:8000/SMBSanalyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                "texts": [ content ]
            })
        }).then(data => data.json()).then(async data => {
            console.log("Data: ", data);

            // Send over to the popup script:
            /**
             * 1. The website we're running this from (website)
             * 2. The result of our data analysis (data)
             */
            // const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
            console.log("Data sending message");
            /*const response = await chrome.runtime.sendMessage({
                action: "message",
                website: window.location.href,
                data
            });*/
            res(data);
        })
    });
}

document.body.addEventListener("load", () => {
    setTimeout(() => {
        getContent();
    }, 3000);
})

window.addEventListener("load", () => {
    getContent();
})

chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if(request.render) {
            // Once we get a message that we opened the popup,
            // send back the analysis we get from the server
            getContent()
            .then(data => {
                sendResponse({
                    data,
                    website: window.location.hostname
                })
            });
        }
        return true;
    }
);