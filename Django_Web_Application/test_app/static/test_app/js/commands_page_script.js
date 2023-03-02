
'use strict';
        
        
const heading = document.querySelector("h3");
heading.textContent = "CLICK HERE TO START";
document.body.addEventListener("click", init);

let ws_url = 'ws://'+window.location.host+'/ws/socket-server-commands/';
const socket = new WebSocket(ws_url);
socket.binaryType = "arraybuffer";

//const context = new AudioContext({sampleRate: 16000,});
const context = new AudioContext();
async function init () {
    if (context.state === 'suspended') {
        await context.resume();
    }
    /*
    const micStream = await navigator.mediaDevices.getUserMedia({
        audio:  {
            //channelCount: 1,
            echoCancellation: false,
            autoGainControl: false,
            noiseSuppression: false,
            latency: 0,
        },
    });*/
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
    const visualizerCallback = visualizer(recordingNode.port);
    recordingNode.port.onmessage = (event) => {
        
        recordingCallback(event);
        visualizerCallback(event);
    };

    visCommandSetup();  
    
}

function sleep(ms){
    return new Promise(resolve => setTimeout(resolve,ms));
}



function handleRecording(recording_port, recording_properties){

    //let mfcc_array = new Array(81).fill(new Float32Array(128));
    //const transpose = matrix => matrix[0].map((col, i) => matrix.map(row => row[i]));

    const recordingEventCallback = async (event) => {
        /*
        if (event.data.message === "CONTINUE_CREATE_BUFFER"){
            //console.log(Meyda.extract("mfcc",event.data.sample));
            mfcc_array[event.data.mfcc_count] = Meyda.extract("mfcc", event.data.sample);
        }
        */
        if(event.data.message === "MAX_BUFFER_LENGTH" ){
            //mfcc_array[event.data.mfcc_count] = Meyda.extract("mfcc", event.data.sample);
            //console.log(event.data);
            //var arr = new Uint8Array([-1, -0.5, 0.7, 1]).buffer;
            console.log(event.data);
            console.log(event.data.buffer_array[0])
            socket.send(event.data.buffer_array[0].buffer);
            //socket.send(arr);
            /*
            socket.send(JSON.stringify({
                message: event.data.message,
                array: event.data.buffer_array[0],
            }));
            */
            //tensor_arr = tf.tensor(transpose(mfcc_array)).print();
            //console.log(tensor_arr);

        }
        //arr = undefined;
        
        
        
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
            canvasCtx.fillStyle = "rgb(200,200,200)"; 
            canvasCtx.fillRect(0,0,width,height);
            canvasCtx.lineWidth = 32;
            for (let i = 0; i < 20; i++){
                
                let freq_val = (gain[i*300])*2;//*Math.exp(1/max);
                let freq_height = Math.round((height/2) * freq_val);
                canvasCtx.beginPath();
                canvasCtx.strokeStyle = "rgb("+(255-i*12).toString()+", 0, "+(i*12).toString()+")";
                canvasCtx.moveTo(i*64, height/2);
                canvasCtx.lineTo(i*64, height/2-freq_height);
                canvasCtx.closePath();
                canvasCtx.stroke();
                if(i === 19){
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

async function visCommandSetup(){
    const canvas = document.querySelector(".command_visualizer");
    const canvasCtx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    canvasCtx.fillStyle = "rgb(200,200,200)"; 
    canvasCtx.fillRect(0,0,width,height);
    canvasCtx.lineWidth = 32;
    let warr = [];
    let xCord = 0;
    let yCord = 0;
    let direction = 'right';
    let color_count = 0;
    let step = 5;

    for (let i = 1; i<=10000;i++){
        
        canvasCtx.fillStyle = "rgb("+(255-color_count*51).toString()+", 0, "+(color_count*51).toString()+")"; 
        if (direction === 'right'){
            if(xCord+step+width/5<=width){
                xCord += step;
                //console.log(xCord);
            }
        }
        if (direction === 'down'){
            if(yCord+step+height/5<=height){
                yCord += step;
            }
        }
        if (direction === 'left'){
            if(xCord-step>=0){
                xCord -= step;
            }
        }
        if (direction === 'up'){
            if(yCord-step>=0){
                yCord -= step;
            }
        }
        if (direction === 'backward'){
                console.log('!!!!!!!!!!!!!!!!!!BACWARD!!!!!!!!!!!!111')
                //window.location.href = '#';
                document.location.href="/";
        }
        if (color_count >= 5){
            color_count = 0;
        } else{
            color_count ++;
        }
        
        canvasCtx.fillRect(xCord,yCord,width/5,height/5);
        

        //draw()

        await sleep(250);
        
        socket.onmessage = function(event){
            var data = JSON.parse(event.data);
            document.querySelector('#command').innerText = data.command;
            warr.push(data.command); 

        }

        if (warr[warr.length-1] != undefined){
            direction = warr[warr.length-1];
        }

        
    }

}
