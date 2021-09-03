let imgTxt = null;
let dataset=null;
$("#result").hide();
//("#success").hide();
const dropArea = document.querySelector(".drag-area"),
dragText = dropArea.querySelector("#header"),
button = dropArea.querySelector("#button1"),
input = dropArea.querySelector("input");
let file; 
button.onclick = ()=>{
  input.click(); 
}
input.addEventListener("change", function(){

  file = this.files[0];
  dropArea.classList.add("active");
  showFile(); 
});

dropArea.addEventListener("dragover", (event)=>{
  event.preventDefault();
  dropArea.classList.add("active");
  dragText.textContent = "Release to Upload File";
});

dropArea.addEventListener("dragleave", ()=>{
  dropArea.classList.remove("active");
  dragText.textContent = "Drag & Drop to Upload File";
});

dropArea.addEventListener("drop", (event)=>{
  event.preventDefault(); 

  file = event.dataTransfer.files[0];
  showFile(); 
});
function showFile(){
  let fileType = file.type; 
  let validExtensions = ["image/jpeg", "image/jpg", "image/png"]; 
  if(validExtensions.includes(fileType)){
    let fileReader = new FileReader();
    fileReader.onload = ()=>{
      let fileURL = fileReader.result; 

      let imgTag = `<img src="${fileURL}" alt="image">`; 
      imgTxt=fileURL;
      dropArea.innerHTML = imgTag; 
    }
    fileReader.readAsDataURL(file);
  }else{
    alert("This is not an Image File!");
    dropArea.classList.remove("active");
    dragText.textContent = "Drag & Drop to Upload File";
  }
}

document.getElementById("button").addEventListener("click", process);

function process(){
    let url = "http://127.0.0.1:5501/detect";

        $.post(url, {imagedata : imgTxt},
            function(data) {
              if(data[1]<0)
                { 
                  $("#resultHolder").show();
                  $("#resultHolder").append("Image rejected with a FitMent score of :" + data[1] +" because "+data[0]);
                  dropArea.classList.remove("active");
                  dragText.textContent = "Drag & Drop to Upload File";
                }
              else if(data[1]>0)
                {
                  $("#resultHolder").show();
                  $("#resultHolder").append("Image accepted with a FitMent score of :" + data[1] );
                }
              else
                {
                  $("#resultHolder").show();
                  $("#resultHolder").append("Error Processing Image" );
                }
            });
}

