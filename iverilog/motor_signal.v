module motor_signal (
    input clk,
    input rst,
    input [1:0] mode,
    output reg signed [15:0] signal
);

reg signed [15:0] sine_lut [0:15];
reg [3:0] idx;

initial begin
    sine_lut[0]=0;   sine_lut[1]=383;
    sine_lut[2]=707; sine_lut[3]=923;
    sine_lut[4]=1000;sine_lut[5]=923;
    sine_lut[6]=707; sine_lut[7]=383;
    sine_lut[8]=0;   sine_lut[9]=-383;
    sine_lut[10]=-707; sine_lut[11]=-923;
    sine_lut[12]=-1000;sine_lut[13]=-923;
    sine_lut[14]=-707; sine_lut[15]=-383;
end

always @(posedge clk) begin
    if (rst) begin
        idx <= 0;
        signal <= 0;
    end else begin
        idx <= idx + 1;

        case (mode)
            2'b00: signal <= sine_lut[idx];           // Healthy
            2'b01: signal <= sine_lut[idx] + 200;     // Bearing
            2'b10: signal <= sine_lut[idx] + (idx*20);// Rotor
            2'b11: signal <= sine_lut[idx] - 300;     // Stator
        endcase
    end
end
endmodule
