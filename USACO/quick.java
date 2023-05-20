import java.util.*;
import java.io.*;
import java.lang.*;

public class quick
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("quick.in"));
      String b = in.next();
      String t = in.next();
      in.close();
      
      for(int i = 0; i < b.length(); i++)
      {
         String temp = new String(b);
         if(b.charAt(i) == '1') temp = temp.substring(0, i) + '0' + temp.substring(i + 1);
         else temp = temp.substring(0, i) + '1' + temp.substring(i + 1);
         int n1 = convertB(temp);
         
         
         for(int j = 0; j < t.length(); j++)
         {
            int n2 = 0, n3 = 0;
            String temp2 = new String(t);
            if(t.charAt(j) == '1')
            {
               temp2 = temp2.substring(0, j) + '0' + temp2.substring(j + 1);
               n2 =  convertT(temp2);
               temp2 = temp2.substring(0, j) + '2' + temp2.substring(j + 1);
               n3 = convertT(temp2);
            }
            else if(t.charAt(j) == '0')
            {
               temp2 = temp2.substring(0, j) + '1' + temp2.substring(j + 1);
               n2 =  convertT(temp2);
               temp2 = temp2.substring(0, j) + '2' + temp2.substring(j + 1);
               n3 = convertT(temp2);
            }
            else
            {
               temp2 = temp2.substring(0, j) + '1' + temp2.substring(j + 1);
               n2 =  convertT(temp2);
               temp2 = temp2.substring(0, j) + '0' + temp2.substring(j + 1);
               n3 = convertT(temp2);
            }
            
            if(n1 == n2 || n1 == n3)
            {
               System.out.println(n1);
               System.exit(0);
            }
         }
      }
   }
   
   public static int convertT(String s)
   {
      int p = 1, result = 0;
      for(int i = s.length() - 1; i >= 0; i--)
      {
         result += (s.charAt(i) - '0') * p;
         p *= 3;
      }
      
      return result;
   }

   public static int convertB(String s)
   {
      int p = 1, result = 0;
      for(int i = s.length() - 1; i >= 0; i--)
      {
         result += (s.charAt(i) - '0') * p;
         p *= 2;
      }
      
      return result;
   }
}