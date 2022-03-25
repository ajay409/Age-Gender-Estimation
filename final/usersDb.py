import joblib
class UsersDb:
	def __init__(self):
		try:
			self.users = joblib.load('users.pkl')
		except Exception as e:
			print(e)
			self.users = {}

	def addUser(self,user):
		if user.get('name') not in self.users:
			self.users[user.get('name')] = user
			self.save()
			return 'success'
		else:
			return 'user already exists'

	def getUser(self,name):
		try:
			user = self.users[name]
			return user
		except Exception as e:
			print(e)

		return False

	def save(self):
		joblib.dump(self.users,'users.pkl')
