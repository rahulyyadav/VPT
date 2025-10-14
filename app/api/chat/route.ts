import { NextRequest, NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import path from 'path'

const execAsync = promisify(exec)

export async function POST(request: NextRequest) {
  try {
    const { message } = await request.json()

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      )
    }

    // Call Python script to get model response
    try {
      const scriptPath = path.join(process.cwd(), 'model_inference.py')
      const command = `python3 "${scriptPath}" "${message.replace(/"/g, '\\"')}"`
      
      const { stdout, stderr } = await execAsync(command, {
        timeout: 30000, // 30 second timeout
        cwd: process.cwd()
      })

      if (stderr) {
        console.error('Python script stderr:', stderr)
      }

      const result = JSON.parse(stdout.trim())
      return NextResponse.json({ response: result.response })
      
    } catch (pythonError) {
      console.error('Error calling Python model:', pythonError)
      
      // Fallback response if model fails
      let fallbackResponse = "I'm having trouble accessing my model right now. "
      
      if (message.toLowerCase().includes('vit')) {
        fallbackResponse += "However, I can tell you that VIT (Vellore Institute of Technology) is located in Vellore, Tamil Nadu, India, with additional campuses in Chennai, Bhopal, and Amravati."
      } else {
        fallbackResponse += "Please try again in a moment."
      }
      
      return NextResponse.json({ response: fallbackResponse })
    }

  } catch (error) {
    console.error('Error in chat API:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
