read -p "Are you sure? " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
	rm -r cached/*
	touch cached/ledger.json
	echo {} > cached/ledger.json
	echo "Reset"
else
	echo "Aborted"
fi
