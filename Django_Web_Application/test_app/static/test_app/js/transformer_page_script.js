document.body.addEventListener("click", init);
const context = new AudioContext();

var socket = new WebSocket('ws://localhost:8000/ws/socket-server-transformer/');
socket.binaryType = "arraybuffer";

async function init () {

    if (context.state === 'suspended') {
        await context.resume();
    }

    const micStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false,
    });

    const micSourceNode = await context.createMediaStreamSource(micStream);

    const recordingProperties = {
        numberOfChannels: micSourceNode.channelCount,
        sampleRate: context.sampleRate,
        bufferLength: 4608,
        visualizeBufferLength: Math.round((context.sampleRate/8)/128)*128,
    };
    console.log(recordingProperties);

    const recordingNode = await setupRecordingWorkletNode(recordingProperties);

    console.log(recordingNode);

    micSourceNode
    .connect(recordingNode)
    //.connect(monitorNode)
    .connect(context.destination);

    const recordingCallback = handleRecording(recordingNode.port);
    const visualizerCallback = visualizer(recordingNode.port);
    recordingNode.port.onmessage = (event) => {
        //console.log(event.data);
        recordingCallback(event);
        visualizerCallback(event);
    }

}

async function setupRecordingWorkletNode(recordingProperties) {
    
    await context.audioWorklet.addModule('/static/test_app/js/recorder_worklet_transformer.js');

    const WorkletRecordingNode = new AudioWorkletNode(
        context,
        'recorder_worklet',
        {
            processorOptions: recordingProperties,
        },
    );

    return WorkletRecordingNode;
}

function handleRecording(recording_port, recording_properties){


    const recordingEventCallback = async (event) => {

        if(event.data.message === "MAX_BUFFER_LENGTH" ){
            socket.send(event.data.buffer_array[0].buffer);
        }            
        
    };
    return recordingEventCallback;

}

function visualizer(recording_port) {
    // Set up canvas context for visualizer
    const canvas = document.querySelector(".visualizer");
    const canvasCtx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const visualizerEventCallback = async (event) => {
        //MAX_BUFFER_LENGTH MAX_VISUALIZER_BUFFER_LENGTH
        if(event.data.message === "MAX_VISUALIZER_BUFFER_LENGTH"){
            draw(event);
        }
    };
    function draw(event) {
        try{
            var gain = event.data.buffer_array[0];
            canvasCtx.fillStyle = "rgb(255,255,255)"; 
            canvasCtx.fillRect(0,0,width,height);
            canvasCtx.lineWidth = 4;
            for (let i = 0; i < 50; i++){
                
                let freq_val = (gain[i*120])*2;//*Math.exp(1/max);
                let freq_height = Math.round((height/2) * freq_val);
                canvasCtx.beginPath();
                canvasCtx.strokeStyle = "rgb(0,0,0)";//"rgb("+(255-i*12).toString()+", 0, "+(i*12).toString()+")";
                canvasCtx.moveTo(i*10, height/2);
                canvasCtx.lineTo(i*10, height/2-freq_height);
                canvasCtx.closePath();
                canvasCtx.stroke();
                if(i === 49){
                    //console.log('trash');
                    gain = null;
                    freq_val=null;
                    freq_height=null;
                    //max=null;
                };

            }
        } catch {
            gain = null;
            //freq_val=null;
            //freq_height=null; 
        }

        requestAnimationFrame(draw);
    }
    return visualizerEventCallback;
}

socket.onmessage = function(event){
    var data = JSON.parse(event.data);
    //document.querySelector('#command').innerText = data.command;
    //warr.push(data.command); 
    console.log(data);
    if (data.message === 'loading_model') {
        document.querySelector('#output').innerText = "Please, wait, model is loading";
    }
    if (data.message === 'model_is_ready') {
        document.querySelector('#output').innerText = "Model is ready, please, click to start.";
    }
    if (data.message === 'decoded_result') {
        document.querySelector('#output').innerText = data.decoded_result;
    }
}