class RecorderProcessor extends AudioWorkletProcessor{
    constructor(options) {

        super();
        
        this.sample_rate = 0;
        this.buffer_length = 0;
        this.number_of_channels = 0;
        //this.visualizer_bufferLength = 0;

        if (options && options.processorOptions) {
            const {
                numberOfChannels,
                sampleRate,
                bufferLength,
                visualizeBufferLength,
            } = options.processorOptions;
    
            this.sample_rate = sampleRate;
            this.buffer_length = bufferLength;
            this.number_of_channels = 1//numberOfChannels;
            this.visualizer_bufferLength = visualizeBufferLength;
        }
        this._recording_buffer = new Array(this.number_of_channels)
            .fill(new Float32Array(this.buffer_length));
        this.visualizer_recording_buffer = new Array(this.number_of_channels)
            .fill(new Float32Array(this.visualizer_bufferLength));
        //this.border_recording_buffer = new Float32Array();
        //this.current_border_Length = 0;
        this.current_bufferLength = 0;
        this.current_visualizer_bufferLength = 0;
        this.mfcc_bool = false;
        //this.type_inp = null;
        //this.border_bool = false;
        //this.mfcc_count = 0;
        
    }

    process(inputs, outputs){
        for (let input = 0; input < 1; input++) {
            for (let channel = 0; channel < this.number_of_channels; channel++) {//this.number_of_channels; channel++) {
                for (let sample = 0; sample < inputs[input][channel].length; sample++) {
                    const current_sample = inputs[input][channel][sample];
                    if(this.mfcc_bool === false){
                        this._recording_buffer[channel][sample] = current_sample;
                        if(Math.abs(current_sample) >= 0.15){
                            //this.type_inp = inputs[input][channel][sample]
                            this.mfcc_bool = true;
                            //for(let i = 0; i < this.border_recording_buffer.length; i++){
                            //    this._recording_buffer[channel][i] = this.border_recording_buffer[i];
                            //}
                            //this.current_bufferLength = this.border_recording_buffer.length;
                            //this.port.postMessage({
                            //    message: 'START_CREATE_BUFFER',
                            //});
                        }
                    }
                    if(this.mfcc_bool === true){
                        this._recording_buffer[channel][this.current_bufferLength+sample] = current_sample;
                    }/* else {
                        this.border_recording_buffer[this.current_border_Length+sample] = current_sample;
                    }*/
                    this.visualizer_recording_buffer[channel][this.current_visualizer_bufferLength+sample] = current_sample; 
                    outputs[input][channel][sample] = current_sample;
                }

            }
        }
        if (this.mfcc_bool === true){
            if(this.current_bufferLength+128 < this.buffer_length){
                /*
                this.port.postMessage({
                    message: 'CONTINUE_CREATE_BUFFER',
                    sample: outputs[0][0],
                    mfcc_count: this.mfcc_count,
                });
                */
                this.current_bufferLength += 128;
                //this.mfcc_count += 1;
            } else {
                this.port.postMessage({
                    message: 'MAX_BUFFER_LENGTH',
                    buffer_array: this._recording_buffer,
                    //sample: outputs[0][0],
                    //mfcc_count: this.mfcc_count,
                });
                this.mfcc_bool = false;
                this.current_bufferLength = 0;
                this._recording_buffer = new Array(this.number_of_channels)
                    .fill(new Float32Array(this.buffer_length));
                //this.mfcc_count = 0;
            }
        }/* else {
            if (this.current_border_Length+128 < 2048){
                this.current_border_Length += 128;
            } else {
                this.current_border_Length = 0;
                this.border_recording_buffer = null;
                this.border_recording_buffer = new Float32Array();
            }
        }*/

        /*
        this.port.postMessage({
            message: 'MAX_BUFFER_LENGTH',
            recording_length: this.current_bufferLength + 128,
            buffer_array: this._recording_buffer,

        });
        this.current_bufferLength = 0;
        this._recording_buffer = null;
        this._recording_buffer = new Array(this.number_of_channels)
                .fill(new Float32Array(this.buffer_length));
        return true;
        */

       
        if (this.current_visualizer_bufferLength + 128 < this.visualizer_bufferLength){
            this.current_visualizer_bufferLength += 128;
        } else {
            this.port.postMessage({
                message: 'MAX_VISUALIZER_BUFFER_LENGTH',
                recording_length: this.current_visualizer_bufferLength + 128,
                buffer_array: this.visualizer_recording_buffer,
            });
            this.current_visualizer_bufferLength = 0;
            this.visualizer_recording_buffer = null;
            this.visualizer_recording_buffer = new Array(this.number_of_channels)
                .fill(new Float32Array(this.visualizer_bufferLength));
        }
        
        /*
        if(this.current_bufferLength + 128 < this.buffer_length){
            this.current_bufferLength += 128;
        } else {
            this.port.postMessage({
                message: 'MAX_BUFFER_LENGTH',
                recording_length: this.current_bufferLength + 128,
                buffer_array: this._recording_buffer,

            });

            this.current_bufferLength = 0;
            //this._recording_buffer = null;
            //this._recording_buffer = new Array(this.number_of_channels)
            //    .fill(new Float32Array(this.buffer_length));
        }
        */
        return true;
        
    }
}

registerProcessor("recorder_worklet", RecorderProcessor);