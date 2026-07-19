import { NextRequest, NextResponse } from 'next/server'

export async function GET(req: NextRequest) {
  const { searchParams } = req.nextUrl
  const city = searchParams.get('city') ?? ''
  const state = searchParams.get('state') ?? ''
  const taxonomy = searchParams.get('taxonomy') ?? ''
  const limit = searchParams.get('limit') ?? '10'

  const params = new URLSearchParams({
    version: '2.1',
    taxonomy_description: taxonomy,
    city: city.toUpperCase(),
    state: state.toUpperCase(),
    limit,
    enumeration_type: 'NPI-1',
  })

  try {
    const res = await fetch(`https://npiregistry.cms.hhs.gov/api/?${params}`, {
      headers: { 'Accept': 'application/json' },
      next: { revalidate: 300 }, // cache 5 min
    })
    if (!res.ok) {
      return NextResponse.json({ error: `NPI API returned ${res.status}` }, { status: 502 })
    }
    const data = await res.json()
    return NextResponse.json(data)
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 })
  }
}
