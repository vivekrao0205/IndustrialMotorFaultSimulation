module feature_extract (
    input clk,
    input rst,
    input signed [15:0] signal,
    output reg [31:0] rms
);

reg [31:0] sum_sq;
reg [9:0] count;

always @(posedge clk) begin
    if (rst) begin
        sum_sq <= 0;
        count <= 0;
        rms <= 0;
    end else begin
        sum_sq <= sum_sq + (signal * signal);
        count <= count + 1;

        if (count == 1023) begin
            rms <= sum_sq >> 10; // approx RMS
            sum_sq <= 0;
            count <= 0;
        end
    end
end
endmodule
