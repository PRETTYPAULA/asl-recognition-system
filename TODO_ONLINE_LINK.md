# Online link (public URL) for the Flask app

This project is a Flask server in `app.py` (routes: `/`, `/dashboard`, `/learning`, `/detection`, `/video_feed`, `/predict`).

To “make it online” (share a link), you need a **tunnel** that exposes your local port `5000` to the internet.

## Option 1: Cloudflare Tunnel (recommended)

1) Install `cloudflared`.
2) Login:
- `cloudflared tunnel login`
3) Create a tunnel:
- `cloudflared tunnel create asl-app`
4) Create a `config.yml` (example):

```yml
# config.yml
# (replace placeholders)
tunnel: <TUNNEL_UUID>
credentials-file: <PATH_TO_credentials_json>

ingress:
  - hostname: <YOUR_PUBLIC_HOSTNAME>   # e.g. asl-app.example.com
    service: http://localhost:5000
  - service: http_status:404
```

5) Run:
- `cloudflared tunnel run asl-app --config config.yml`

6) Copy the `https://<hostname>` link shown in the terminal.

## Option 2: ngrok

1) Install ngrok.
2) Run:
- `ngrok http 5000`
3) Copy the **https** public URL.

## “Free website convert” summary (what to do)
- You do **not** need to modify `app.py` to serve HTTPS.
- Run a tunnel (Cloudflare Tunnel or ngrok) in front of your Flask server.
- Use the **https://** URL printed by the tunnel in your browser.



## Webcam note (important)
`/video_feed` streams your local webcam. Viewers opening the public link will see the webcam from the device running the server.

