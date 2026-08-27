import os
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

mongo_uri = os.environ.get("MONGO_URI")
client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
db = client[os.environ.get("MONGO_DB_NAME", "skillnaav-land")]

applications_col = db["applications"]
shortlist_col = db["shortlist"]

def main():
    shortlisted_apps = list(applications_col.find({"status": "Shortlisted"}))
    print(f"Found {len(shortlisted_apps)} applications with status 'Shortlisted'")
    
    fixed_count = 0
    for app in shortlisted_apps:
        internship_id = app.get("internshipId")
        resume_url = app.get("resumeUrl")
        
        # Check if it actually exists in shortlist for this internship
        is_in_shortlist = shortlist_col.find_one({
            "internship_id": internship_id,
            "resumeUrl": resume_url
        })
        
        if not is_in_shortlist:
            # Revert to Applied
            applications_col.update_one(
                {"_id": app["_id"]},
                {"$set": {"status": "Applied"}}
            )
            print(f"Reset status for {app.get('userEmail')} on internship {internship_id}")
            fixed_count += 1
            
    print(f"Successfully fixed {fixed_count} mismatched applications.")

if __name__ == "__main__":
    main()
