module tb_motor_fault;

reg clk;
reg rst;
reg [1:0] mode;

wire signed [15:0] signal;
wire [31:0] rms;
wire [1:0] fault;

motor_signal U1(.clk(clk), .rst(rst), .mode(mode), .signal(signal));
feature_extract U2(.clk(clk), .rst(rst), .signal(signal), .rms(rms));
classifier U3(.rms(rms), .fault(fault));

always #5 clk = ~clk;

initial begin
    // INITIALIZE EVERYTHING
    clk  = 0;
    rst  = 1;
    mode = 2'b00;

    $dumpfile("motor.vcd");
    $dumpvars(0, tb_motor_fault);

    // HOLD RESET PROPERLY
    #50;
    rst = 0;

    // APPLY MODES SLOWLY
    #2000 mode = 2'b00;  // Healthy
    #2000 mode = 2'b01;  // Bearing
    #2000 mode = 2'b10;  // Rotor
    #2000 mode = 2'b11;  // Stator

    #2000 $finish;
end

endmodule


