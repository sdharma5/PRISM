export interface NPIProvider {
  npi: string
  name: string
  credential: string
  gender: string
  specialty: string
  taxonomyCode: string
  address: string
  city: string
  state: string
  zip: string
  phone: string
  mapsUrl: string
}

export type Specialty = 'obgyn' | 'reproductive_endo' | 'endocrinology' | 'dermatology'

const TAXONOMY_MAP: Record<Specialty, { code: string; label: string; description: string }> = {
  obgyn: {
    code: '207V00000X',
    label: 'OB-GYN',
    description: 'Obstetrics & Gynecology',
  },
  reproductive_endo: {
    code: '207VE0102X',
    label: 'Reproductive Endocrinologist',
    description: 'Reproductive Endocrinology',
  },
  endocrinology: {
    code: '207RE0101X',
    label: 'Endocrinologist',
    description: 'Endocrinology',
  },
  dermatology: {
    code: '207N00000X',
    label: 'Dermatologist',
    description: 'Dermatology',
  },
}

export function specialtyLabel(s: Specialty) {
  return TAXONOMY_MAP[s].label
}

export function specialtyDescription(s: Specialty) {
  return TAXONOMY_MAP[s].description
}

export async function searchProviders(
  city: string,
  state: string,
  specialty: Specialty,
  limit = 10,
): Promise<NPIProvider[]> {
  const { code } = TAXONOMY_MAP[specialty]
  const params = new URLSearchParams({
    city: city.trim(),
    state: state.trim().toUpperCase(),
    taxonomy: TAXONOMY_MAP[specialty].description,
    limit: String(limit),
  })

  const res = await fetch(`/api/npi?${params}`)
  if (!res.ok) throw new Error(`NPI API error: ${res.status}`)
  const data = await res.json()
  if (data.error) throw new Error(data.error)

  if (!data.results) return []

  return (data.results as NPIResult[]).map((r) => {
    const basic = r.basic ?? {}
    const taxonomy = r.taxonomies?.find((t) => t.primary) ?? r.taxonomies?.[0]
    const addr = r.addresses?.find((a) => a.address_purpose === 'LOCATION') ?? r.addresses?.[0]

    const firstName = basic.first_name ?? ''
    const lastName = basic.last_name ?? ''
    const credential = basic.credential ?? ''
    const name = `${firstName} ${lastName}`.trim() || 'Unknown Provider'

    const address1 = addr?.address_1 ?? ''
    const address2 = addr?.address_2 ?? ''
    const city2 = addr?.city ?? city
    const state2 = addr?.state ?? state
    const zip = addr?.postal_code?.slice(0, 5) ?? ''
    const phone = addr?.telephone_number ?? ''

    const fullAddress = [address1, address2].filter(Boolean).join(', ')
    const mapsQuery = encodeURIComponent(`${name} ${fullAddress} ${city2} ${state2}`)

    return {
      npi: r.number,
      name,
      credential,
      gender: basic.gender === 'M' ? 'Male' : basic.gender === 'F' ? 'Female' : '',
      specialty: taxonomy?.desc ?? specialtyDescription(specialty),
      taxonomyCode: taxonomy?.code ?? code,
      address: fullAddress,
      city: city2,
      state: state2,
      zip,
      phone,
      mapsUrl: `https://www.google.com/maps/search/?api=1&query=${mapsQuery}`,
    }
  })
}

// NPI API response shape (partial)
interface NPIResult {
  number: string
  basic?: {
    first_name?: string
    last_name?: string
    credential?: string
    gender?: string
  }
  taxonomies?: Array<{ code: string; desc: string; primary: boolean }>
  addresses?: Array<{
    address_purpose: string
    address_1: string
    address_2?: string
    city: string
    state: string
    postal_code: string
    telephone_number: string
  }>
}
