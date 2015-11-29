import argparse
import RotationalScattering as rotscat


parser = argparse.ArgumentParser(description='Tool for simulating spin-echo measurements of rotational molecular diffusion.')
parser.add_argument('--type', type=int, help='diffusion type:\n 0 discrete (Markov / Monte Carlo model) \n 1 continuous (Langevin model)')


args = parser.parse_args()


#check arguments make sense
if args.type == 0:
    rotscat.discrete_Markov_scattering()
elif args.type == 1:
    rotscat.continuous_Langevin_scattering()
else:
    print("Error: invalid model choice; must be 0 (discrete) or 1 (continuous).")



