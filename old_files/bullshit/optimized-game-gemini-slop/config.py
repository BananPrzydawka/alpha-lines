
height = 10
width = 16

# to test model size brakedown:
# from torchinfo import summary
# summary(model, input_size=(1, 7, 10, 16))

filters = 128               # 256 in alpha zero
bottleneck = 32             # 32 in lc0
resblock_number = 20        # 40 in alpha zero

policy_filters = 40         # 80 in lc0
value_fc = 256
error_fc = 128
point_fc = 256