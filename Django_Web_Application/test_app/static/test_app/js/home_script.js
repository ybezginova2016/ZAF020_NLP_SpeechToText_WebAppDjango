
document.body.addEventListener("click", init);
//window.onload = init;

var div_select = document.getElementById("div-select");
div_select.onmouseover = function(){
    this.setAttribute("style","background-color: black");
}
div_select.onmouseout = function(){
    this.setAttribute("style","background-color: white");
}

div_select.addEventListener("click", function onClick(event){
    this.setAttribute("style","background-color: rgb(201, 218, 235)");
    //document.body.style.backgroundColor = "white";
});

var div_select2 = document.getElementById("div-select2");
div_select2.onmouseover = function(){
    this.setAttribute("style","background-color: black");
}
div_select2.onmouseout = function(){
    this.setAttribute("style","background-color: white");
}
div_select2.addEventListener("click", function onClick(event){
    this.setAttribute("style","background-color: rgb(201, 218, 235)");
    //document.body.style.backgroundColor = "white";
});

let ws_url = 'ws://'+window.location.host+'/ws/socket-server-commands/';
const socket = new WebSocket(ws_url);
socket.binaryType = "arraybuffer";

const context = new AudioContext();

//getAudioContext().resume();

async function init () {

    if (context.state === 'suspended') {
        await context.resume();
    }

    const micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
            channelCount: 1,
            //sampleRate: 16000,
        },
        video: false,
    });

    const micSourceNode = await context.createMediaStreamSource(micStream);

    const recordingProperties = {
        numberOfChannels: micSourceNode.channelCount,
        sampleRate: context.sampleRate,
        bufferLength: context.sampleRate,//Math.round((context.sampleRate/4)/128)*128 + 128,
        visualizeBufferLength: Math.round((context.sampleRate/8)/128)*128,
    };
    console.log(recordingProperties);
    
    const recordingNode = await setupRecordingWorkletNode(recordingProperties);

    console.log(recordingNode);

    micSourceNode
    .connect(recordingNode)
    //.connect(monitorNode)
    .connect(context.destination);

    const recordingCallback = handleRecording(recordingNode.port, recordingProperties);
    recordingNode.port.onmessage = (event) => {
        recordingCallback(event);
    };

}
function handleRecording(recording_port, recording_properties){

    const recordingEventCallback = async (event) => {

        if(event.data.message === "MAX_BUFFER_LENGTH" ){

            console.log(event.data);
            console.log(event.data.buffer_array[0])
            socket.send(event.data.buffer_array[0].buffer);

        }
        
        
        
    };
    return recordingEventCallback;
}

async function setupRecordingWorkletNode(recordingProperties) {
    await context.audioWorklet.addModule('/static/test_app/js/recorder_worklet.js');

    const WorkletRecordingNode = new AudioWorkletNode(
        context,
        'recorder_worklet',
        {
            processorOptions: recordingProperties,
        },
    );

    return WorkletRecordingNode;
}

var last_command;

socket.onmessage = function(event){
    var data = JSON.parse(event.data);
    if (data.command == 'left'){
        last_command = 'left';
        div_select.style.backgroundColor = "black";
        div_select2.style.backgroundColor = "white";
    }
    if (data.command == 'right'){
        last_command = 'right'
        div_select.style.backgroundColor = "white";
        div_select2.style.backgroundColor = "black";
    }
    if (data.command == 'go'){
        console.log(last_command);
        if(last_command == 'left'){
            window.location.href = '/commandsrec';
        }
        if(last_command == 'right'){
            window.location.href = '/transformer';
        }
    }
    //document.querySelector('#command').innerText = data.command;
    //warr.push(data.command); 

}

