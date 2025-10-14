# VITGPT - Minimal ChatGPT-like Web App

A clean, minimalist ChatGPT-like interface built with Next.js and TypeScript.

## Features

- Clean, responsive chat interface
- Real-time message display
- TypeScript support
- Modern UI with smooth animations
- Ready for AI model integration

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

- `app/page.tsx` - Main chat interface component
- `app/api/chat/route.ts` - API endpoint for handling chat messages
- `app/globals.css` - Global styles for the chat interface
- `vpt_model (1).pt` - Your trained model (ready for integration)

## Next Steps

The API route currently returns placeholder responses. You can integrate your trained model (`vpt_model (1).pt`) in the `/api/chat/route.ts` file to generate actual AI responses.

## Tech Stack

- Next.js 14
- React 18
- TypeScript
- CSS3
