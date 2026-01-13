module classifier (
    input [31:0] rms,
    output reg [1:0] fault
);

always @(*) begin
    if (rms < 100000)
        fault = 2'b00; // Healthy
    else if (rms < 200000)
        fault = 2'b01; // Bearing
    else if (rms < 300000)
        fault = 2'b10; // Rotor
    else
        fault = 2'b11; // Stator
end
endmodule
