console.log("This is content script doing its thing!");

const searchForLinks = () => {
    // Query for all the content within the page
    const content = document.body.innerText;
    console.log("Content: ", content);

    setTimeout(searchForLinks, 1000);
}

document.body.addEventListener("load", () => {
    searchForLinks();
})

window.addEventListener("load", () => {
    searchForLinks();
})
