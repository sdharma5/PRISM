export interface Insurer {
  id: string
  name: string
  searchUrl: (providerName: string, city: string, state: string) => string
  prefilled: boolean
  note?: string
}

export const INSURERS: Insurer[] = [
  {
    id: 'medicare',
    name: 'Medicare',
    prefilled: true,
    searchUrl: (name, city, state) =>
      `https://www.medicare.gov/care-compare/results?searchType=name&name=${encodeURIComponent(name)}&location=${encodeURIComponent(`${city}, ${state}`)}&radius=25&benefitType=physician`,
  },
  {
    id: 'medicaid',
    name: 'Medicaid',
    prefilled: false,
    note: 'Medicaid directories are managed by each state — your state Medicaid site will have the most accurate list.',
    searchUrl: (_n, _c, state) =>
      `https://www.medicaid.gov/about-us/contact-us/contact-state-page.html`,
  },
  {
    id: 'unitedhealthcare',
    name: 'UnitedHealthcare',
    prefilled: true,
    searchUrl: (name, city, state) =>
      `https://find-a-doctor.uhc.com/results?name=${encodeURIComponent(name)}&location=${encodeURIComponent(`${city}, ${state}`)}`,
  },
  {
    id: 'bcbs',
    name: 'BlueCross BlueShield',
    prefilled: false,
    note: 'BCBS plans vary by state. Select your state plan on their site for the most accurate directory.',
    searchUrl: () => `https://www.bcbs.com/find-a-doctor`,
  },
  {
    id: 'aetna',
    name: 'Aetna',
    prefilled: false,
    searchUrl: () => `https://www.aetna.com/individuals-families/find-a-doctor.html`,
  },
  {
    id: 'cigna',
    name: 'Cigna / Evernorth',
    prefilled: false,
    searchUrl: () => `https://hcpdirectory.cigna.com/web/public/consumer/directory/search`,
  },
  {
    id: 'humana',
    name: 'Humana',
    prefilled: false,
    searchUrl: () => `https://www.humana.com/finder/medical/`,
  },
  {
    id: 'kaiser',
    name: 'Kaiser Permanente',
    prefilled: false,
    searchUrl: () => `https://healthy.kaiserpermanente.org/choose-my-doctor`,
  },
  {
    id: 'molina',
    name: 'Molina Healthcare',
    prefilled: false,
    searchUrl: () => `https://www.molinahealthcare.com/members/com/en-US/hp/findadoctor.aspx`,
  },
  {
    id: 'anthem',
    name: 'Anthem',
    prefilled: false,
    searchUrl: () => `https://www.anthem.com/find-a-doctor/`,
  },
  {
    id: 'other',
    name: 'Other / Unknown',
    prefilled: false,
    note: 'Enter your plan name in the phone script and ask the office to verify coverage.',
    searchUrl: () => `https://www.cms.gov/marketplace/consumers/how-to-apply`,
  },
]

export function getInsurer(id: string): Insurer | undefined {
  return INSURERS.find(i => i.id === id)
}

export const INSURANCE_STORAGE_KEY = 'prism_insurance_plan'

export function loadInsurancePlan(): string {
  if (typeof window === 'undefined') return ''
  return localStorage.getItem(INSURANCE_STORAGE_KEY) ?? ''
}

export function saveInsurancePlan(id: string) {
  if (typeof window === 'undefined') return
  localStorage.setItem(INSURANCE_STORAGE_KEY, id)
}
