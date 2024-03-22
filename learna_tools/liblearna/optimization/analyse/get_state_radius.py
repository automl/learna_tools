def get_state_radius(conv_radius1, conv_radius2, state_rel):
    conv_size1 = 1 + 2 * conv_radius1
    if conv_radius1 == 0:
        conv_size1 = 0

    conv_size2 = 1 + 2 * conv_radius2
    if conv_radius2 == 0:
        conv_size2 = 0

    if conv_size1 != 0:
        min_state_radius = conv_size1 + conv_size1 - 1
        max_state_radius = 32  # FR changed max state radius from 32 to 64, ICLR: LEARNA (32), Meta-LEARNA (29)
        state_radius = int(
            min_state_radius
            + (max_state_radius - min_state_radius) * state_rel
        )
    else:
        min_state_radius = conv_size2 + conv_size2 - 1
        max_state_radius = 32  # FR changed max state radius from 32 to 64, ICLR: LEARNA (32), Meta-LEARNA (29)
        state_radius = int(
            min_state_radius
            + (max_state_radius - min_state_radius) * state_rel
        )
    return state_radius




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--conv1", type=int, help="Conv_size 1")
    parser.add_argument("--conv2", type=int, help="Conv_size 2")
    parser.add_argument("--state_rel", type=float, help="state_radius_relative")

    args = parser.parse_args()

    print(get_state_radius(args.conv1, args.conv2, args.state_rel))
