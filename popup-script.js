const renderSliderGivenElem = (slider, data) => {
  // Sort keys according to their significance
  for(let attr of Object.keys(data)) {
    slider.querySelector(`.${attr}`).setAttribute("style", `width:${data[attr]*100}%`);
    if(data[attr] < 0.10) {
      // If the emotion isn't large enough, don't show text
      slider.querySelector(`.${attr}`).innerHTML = "";
    }
  }
}

const renderSliders = (data) => {
  // (1) Emotion sliders
  const emotionSlider = document.querySelector(".emotion-slider");
  renderSliderGivenElem(emotionSlider, data.dominating_emotions);

  // (2) Political bias
  const politicalSlider = document.querySelector(".politics-slider");
  renderSliderGivenElem(politicalSlider, data.political_bias);

  // (3) Stereotype
  const stereotypeSlider = document.querySelector(".stereotype-slider");
  // Get points to add up to 1
  const stereo = data.stereotype_analysis;
  const total = (Object.keys(stereo).map(e => stereo[e]).reduce((a, b) => a+b));
  let newStereo = data.stereotype_analysis;
  for(let key of Object.keys(newStereo)) {
    newStereo[key] /= total;
  }
  renderSliderGivenElem(stereotypeSlider, newStereo);
}

(async () => {
  const loading = document.querySelector(".loading");
  loading.setAttribute("style", "display: flex;");
  // Show the website being read
  const webElem = document.querySelector(".website");
  // Send a message to the tab we're currently on (where the content-script is running)
  const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
  webElem.innerHTML = (new URL(tab.url)).hostname.replace("www.", "");
  const {data} = await chrome.tabs.sendMessage(tab.id, {render: true});
  
  // Once we get the response, render it!
  renderSliders(data);

  // Fill "content is mostly ..."
  console.log("Data: ", data);
  const negPos = data.sentiment_score;
  const intensity = document.querySelector(".intensity");
  const negPosElem = document.querySelector(".content");
  if(Math.abs(negPos) < 0.9) {
    intensity.innerHTML = "";
  }
  else if(Math.abs(negPos) < 0.9) {
    intensity.innerHTML = "moderately";
  }
  else {
    intensity.innerHTML = "mostly";
  }
  console.log("Intensity: ", negPos);
  negPosElem.innerHTML = Math.abs(negPos) < 0.9 ? "NEUTRAL" : 
    negPos < -0.9 ? "NEGATIVE" : "POSITIVE";
  loading.setAttribute("style", "display: none;");
})();