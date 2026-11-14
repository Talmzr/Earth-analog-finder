"""Run once in YOUR terminal: asks for your Earthdata username/password and saves them to ~/.netrc
(_netrc on Windows) so the scripts can download Earthdata products. Nothing is sent anywhere else."""
import earthaccess

auth = earthaccess.login(strategy="interactive", persist=True)
print("\nEarthdata login OK:", bool(auth.authenticated))
